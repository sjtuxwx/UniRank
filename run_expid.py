# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# Copyright (C) 2025. FuxiCTR Authors. All rights reserved.
# =========================================================================

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import gc
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fuxictr.utils import (
    load_config, set_logger, print_to_json, print_to_list,
    parse_gpu_ids, setup_visible_devices, init_distributed_env, is_main_process
)
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from UniRank_Dataloader import UniRankDataloader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import model_zoo


def attach_distributed_params(params, distributed, rank, local_rank, world_size):
    run_params = dict(params)
    if distributed:
        run_params.update({
            "distributed": True,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size
        })
    else:
        run_params.update({
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1
        })
    return run_params


def make_train_valid_iterators(feature_map, params, distributed, rank, local_rank, world_size):
    train_params = attach_distributed_params(params, distributed, rank, local_rank, world_size)
    return RankDataLoader(feature_map, stage='train', **train_params).make_iterator()


def make_test_iterator(feature_map, params, distributed, rank, local_rank, world_size):
    test_params = attach_distributed_params(params, distributed, rank, local_rank, world_size)
    return RankDataLoader(feature_map, stage='test', **test_params).make_iterator()


def list_rolling_day_files(params):
    rolling_dir = params.get("rolling_data_dir", None)
    if rolling_dir is None:
        train_data = params.get("train_data", None)
        if train_data is None:
            raise ValueError("rolling_train=True requires rolling_data_dir or train_data.")
        rolling_dir = str(Path(train_data).parent)

    rolling_dir = Path(rolling_dir)
    if not rolling_dir.exists():
        raise FileNotFoundError(f"rolling_data_dir does not exist: {rolling_dir}")

    start_day = params.get("rolling_start_day", None)
    end_day = params.get("rolling_end_day", None)
    start_day = str(start_day) if start_day is not None else None
    end_day = str(end_day) if end_day is not None else None

    day_files = []
    for path in sorted(rolling_dir.glob("*.parquet")):
        day = path.stem
        if len(day) != 4 or (not day.isdigit()):
            continue
        if start_day is not None and day < start_day:
            continue
        if end_day is not None and day > end_day:
            continue
        day_files.append((day, str(path)))

    if len(day_files) < 2:
        raise ValueError(
            f"Rolling training needs at least 2 day parquet files, got {len(day_files)} "
            f"from {rolling_dir} with start_day={start_day}, end_day={end_day}."
        )
    return day_files


def build_rolling_pairs(day_files):
    pairs = []
    for idx in range(len(day_files) - 1):
        train_day, train_data = day_files[idx]
        valid_day, valid_data = day_files[idx + 1]
        pairs.append({
            "train_day": train_day,
            "train_data": train_data,
            "valid_day": valid_day,
            "valid_data": valid_data,
            "is_final": idx == len(day_files) - 2
        })
    return pairs


def rolling_checkpoint_path(base_checkpoint, train_day, valid_day):
    base_path = Path(base_checkpoint)
    checkpoint_dir = base_path.parent / "rolling"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = base_path.suffix or ".model"
    return str(checkpoint_dir / f"{base_path.stem}_{train_day}_to_{valid_day}{suffix}")


def run_standard_training(model, feature_map, params, distributed, rank, local_rank, world_size):
    params["data_loader"] = UniRankDataloader

    train_gen, valid_gen = make_train_valid_iterators(
        feature_map, params, distributed, rank, local_rank, world_size
    )

    if distributed and dist.is_initialized():
        dist.barrier()
    model.fit(train_gen, validation_data=valid_gen, **params)

    del train_gen, valid_gen
    gc.collect()

    if params.get("test_data", None):
        if is_main_process(rank):
            logging.info('******** Test evaluation ********')

        test_gen = make_test_iterator(
            feature_map, params, distributed, rank, local_rank, world_size
        )

        if distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()
        model.evaluate(test_gen)

        del test_gen
        gc.collect()


def run_rolling_training(model, feature_map, params, distributed, rank, local_rank, world_size):
    params["data_loader"] = UniRankDataloader

    day_files = list_rolling_day_files(params)
    rolling_pairs = build_rolling_pairs(day_files)
    rolling_epochs = int(params.get("rolling_epochs_per_day", params.get("epochs", 1)))
    original_eval_steps = params.get("eval_steps", None)
    base_checkpoint = model.checkpoint

    if is_main_process(rank):
        logging.info("******** Rolling training enabled ********")
        logging.info("Rolling days: " + " - ".join([x[0] for x in day_files]))
        logging.info("Rolling epochs per day: {}".format(rolling_epochs))

    final_valid_data = rolling_pairs[-1]["valid_data"]
    final_valid_day = rolling_pairs[-1]["valid_day"]
    final_checkpoint = None

    for step, pair in enumerate(rolling_pairs, start=1):
        day_params = dict(params)
        day_params["epochs"] = rolling_epochs
        day_params["train_data"] = pair["train_data"]
        day_params["valid_data"] = pair["valid_data"]
        day_params["test_data"] = None

        model.checkpoint = rolling_checkpoint_path(
            base_checkpoint, pair["train_day"], pair["valid_day"]
        )
        model._eval_steps = original_eval_steps

        if is_main_process(rank):
            logging.info(
                "******** Rolling step {}/{}: train={} valid={} checkpoint={} ********"
                .format(
                    step,
                    len(rolling_pairs),
                    pair["train_day"],
                    pair["valid_day"],
                    model.checkpoint
                )
            )

        if distributed and dist.is_initialized():
            dist.barrier()

        train_gen, valid_gen = make_train_valid_iterators(
            feature_map, day_params, distributed, rank, local_rank, world_size
        )
        model.fit(train_gen, validation_data=valid_gen, **day_params)

        final_checkpoint = model.checkpoint
        del train_gen, valid_gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if distributed and dist.is_initialized():
            dist.barrier()

    if is_main_process(rank):
        logging.info(
            "******** Final rolling evaluation: day={} checkpoint={} ********"
            .format(final_valid_day, final_checkpoint)
        )

    final_params = dict(params)
    final_params["test_data"] = final_valid_data
    test_gen = make_test_iterator(
        feature_map, final_params, distributed, rank, local_rank, world_size
    )

    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
    model.evaluate(test_gen)

    del test_gen
    gc.collect()
    model.checkpoint = base_checkpoint


if __name__ == '__main__':
    """
    单卡:
      python run_expid.py

    DDP 多卡(单机):
      torchrun --standalone --nproc_per_node=2 run_expid.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DIN_KuaiRand_Video_Action', help='The experiment id to run.')
    parser.add_argument('--gpu', type=str, default='1,2', help='GPU ids, e.g. "0" or "0,1,2"; use "-1" for cpu')
    parser.add_argument('--enable_bf16', type=bool, default=True, help='Enable bfloat16 mixed precision training (default: True).')
    args = vars(parser.parse_args())

    try:
        gpu_ids = parse_gpu_ids(args['gpu'])
        setup_visible_devices(gpu_ids)

        distributed, rank, local_rank, world_size, local_world_size = init_distributed_env()

        # 基本合法性检查 + 分布式初始化
        if distributed:
            if len(gpu_ids) == 0:
                raise ValueError("DDP 模式下不能使用 CPU（--gpu -1）")
            if local_world_size != len(gpu_ids):
                raise ValueError(
                    f"LOCAL_WORLD_SIZE({local_world_size}) 与 --gpu 数量({len(gpu_ids)}) 不一致。"
                    f"请保证 torchrun --nproc_per_node 与 --gpu 个数一致。"
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用，无法使用 DDP+NCCL。")

            torch.cuda.set_device(local_rank)

            # 仅在未初始化时初始化，避免重复初始化报错
            if not dist.is_available():
                raise RuntimeError("torch.distributed 不可用。")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
            dist.barrier()
        else:
            # 非 torchrun 模式下，不允许给多个 GPU（避免误以为自动 DDP）
            if len(gpu_ids) > 1:
                raise ValueError(
                    "检测到多个 GPU，但当前不是 torchrun 模式。\n"
                    "请使用: torchrun --standalone --nproc_per_node=<GPU数量> run_expid.py ... --gpu 0,1,..."
                )
            if len(gpu_ids) == 1 and not torch.cuda.is_available():
                raise RuntimeError("指定了 GPU 但 CUDA 不可用。")

        experiment_id = args['expid']
        params = load_config(args['config'], experiment_id)

        # 设备参数：
        # - DDP: local_rank 映射到 CUDA_VISIBLE_DEVICES 内部序号
        # - 单卡GPU: 固定使用可见设备 0
        if len(gpu_ids) == 0:
            params['gpu'] = -1
        else:
            params['gpu'] = local_rank if distributed else 0

        params['distributed'] = distributed
        params['rank'] = rank
        params['local_rank'] = local_rank
        params['world_size'] = world_size

        # bf16 开关：命令行参数覆盖 config 中的同名字段（若存在）
        params['enable_bf16'] = args['enable_bf16']

        if is_main_process(rank):
            set_logger(params)
            logging.info("Params: " + print_to_json(params))
        else:
            logging.getLogger().handlers = []
            logging.basicConfig(level=logging.ERROR)

        # 每个 rank 使用不同 seed 偏移
        seed_everything(seed=params['seed'] + rank)

        data_dir = os.path.join(params['data_root'], params['dataset_id'])
        feature_map_json = os.path.join(data_dir, "feature_map.json")

        if distributed:
            if is_main_process(rank):
                feature_encoder = FeatureProcessor(**params)
                params["train_data"], params["valid_data"], params["test_data"] = \
                    build_dataset(feature_encoder, **params)

            obj_list = [[
                params.get("train_data", None),
                params.get("valid_data", None),
                params.get("test_data", None)
            ]] if is_main_process(rank) else [[None, None, None]]

            dist.broadcast_object_list(obj_list, src=0)
            params["train_data"], params["valid_data"], params["test_data"] = obj_list[0]
            dist.barrier()
        else:
            feature_encoder = FeatureProcessor(**params)
            params["train_data"], params["valid_data"], params["test_data"] = \
                build_dataset(feature_encoder, **params)

        feature_map = FeatureMap(params['dataset_id'], data_dir)
        feature_map.load(feature_map_json, params)
        if is_main_process(rank):
            logging.info("Feature specs: " + print_to_json(feature_map.features))

        model_class = getattr(model_zoo, params['model'])
        model = model_class(feature_map, **params)
        model.model_to_device()

        if distributed:
            ddp_model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=params.get("find_unused_parameters", True)
            )
            # 依赖你前面改过的 rank_model.py
            model.set_ddp_model(ddp_model)

        if is_main_process(rank):
            model.count_parameters()

        if params.get("rolling_train", False):
            run_rolling_training(
                model, feature_map, params, distributed, rank, local_rank, world_size
            )
        else:
            run_standard_training(
                model, feature_map, params, distributed, rank, local_rank, world_size
            )

    finally:
        if distributed and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
