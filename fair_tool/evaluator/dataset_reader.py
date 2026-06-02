from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from UniRank_Dataloader import UniRankDataloader
from fuxictr.pytorch.dataloaders import RankDataLoader

from fair_tool.utils.checkpoint_loader import CheckpointLoader, DatasetName


@dataclass(frozen=True)
class EvaluationDatasetInfo:
    dataset_name: str
    dataset_id: str
    eval_date: str
    eval_path: str
    perday: bool
    streaming: bool


class EvaluationDatasetReader:
    @staticmethod
    def _is_perday_eval_date(eval_date: str) -> bool:
        eval_date_str = str(eval_date).strip()
        return len(eval_date_str) == 4 and eval_date_str.isdigit()

    @staticmethod
    def _load_public_context() -> tuple[dict[str, Any], Path, Path]:
        config, config_path = CheckpointLoader._load_public_config()
        project_root = CheckpointLoader._project_root(config, config_path)
        return config, config_path, project_root

    @staticmethod
    def _resolve_eval_info(
        dataset_name: DatasetName | str,
        eval_date: str,
        streaming: bool,
    ) -> EvaluationDatasetInfo:
        config, config_path, project_root = EvaluationDatasetReader._load_public_context()
        dataset_key = CheckpointLoader.normalize_enum_value(dataset_name)
        perday = EvaluationDatasetReader._is_perday_eval_date(eval_date)
        dataset_id = CheckpointLoader.resolve_dataset_id(dataset_key, perday=perday)

        dataset_config = CheckpointLoader._read_yaml(
            CheckpointLoader._resolve_config_path(config["dataset_config"], config_path)
        )
        if dataset_id not in dataset_config:
            raise ValueError(f"dataset_id={dataset_id!r} is not defined in dataset_config.")

        data_root = Path(dataset_config[dataset_id].get("data_root", "./data/"))
        if not data_root.is_absolute():
            data_root = (project_root / data_root).resolve()

        if perday:
            eval_path = data_root / dataset_id / f"{eval_date}.parquet"
        else:
            eval_date_key = str(eval_date).strip().lower()
            if eval_date_key not in {"train", "valid", "test"}:
                raise ValueError(
                    "Non-perday evaluation only supports eval_date in {'train', 'valid', 'test'}."
                )
            eval_path = data_root / dataset_id / f"{eval_date_key}.parquet"

        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {eval_path}")

        return EvaluationDatasetInfo(
            dataset_name=dataset_key,
            dataset_id=dataset_id,
            eval_date=str(eval_date),
            eval_path=str(eval_path),
            perday=perday,
            streaming=streaming,
        )

    @staticmethod
    def _build_test_iterator(
        dataset_info: EvaluationDatasetInfo,
        feature_map,
        model_id: str,
        batch_size: int | None = None,
        streaming: bool = False,
    ):
        config, config_path, _ = EvaluationDatasetReader._load_public_context()
        params = CheckpointLoader._load_runtime_params(model_id, config, config_path)
        if feature_map.dataset_id != dataset_info.dataset_id:
            raise ValueError(
                f"Model feature_map.dataset_id={feature_map.dataset_id!r} does not match "
                f"evaluation dataset_id={dataset_info.dataset_id!r}."
            )
        params["dataset_id"] = dataset_info.dataset_id
        params["test_data"] = dataset_info.eval_path
        if batch_size is None:
            batch_size = config.get("default_eval_batch_size", 256)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        params["batch_size"] = batch_size
        params["streaming"] = streaming
        params["shuffle"] = False
        params["data_loader"] = UniRankDataloader
        params["distributed"] = False
        params["rank"] = 0
        params["local_rank"] = 0
        params["world_size"] = 1
        return RankDataLoader(feature_map, stage="test", **params).make_iterator()

    @staticmethod
    def load_all(
        dataset_name: DatasetName | str,
        eval_date: str,
        feature_map=None,
        model_id: str | None = None,
        batch_size: int | None = None,
    ):
        dataset_info = EvaluationDatasetReader._resolve_eval_info(
            dataset_name=dataset_name,
            eval_date=eval_date,
            streaming=False,
        )
        if feature_map is None or model_id is None:
            return None, asdict(dataset_info)
        test_gen = EvaluationDatasetReader._build_test_iterator(
            dataset_info=dataset_info,
            feature_map=feature_map,
            model_id=model_id,
            batch_size=batch_size,
            streaming=False,
        )
        return test_gen, asdict(dataset_info)

    @staticmethod
    def stream_batches(
        dataset_name: DatasetName | str,
        eval_date: str,
        feature_map=None,
        model_id: str | None = None,
        batch_size: int | None = None,
    ):
        dataset_info = EvaluationDatasetReader._resolve_eval_info(
            dataset_name=dataset_name,
            eval_date=eval_date,
            streaming=True,
        )
        if feature_map is None or model_id is None:
            return None, asdict(dataset_info)
        test_gen = EvaluationDatasetReader._build_test_iterator(
            dataset_info=dataset_info,
            feature_map=feature_map,
            model_id=model_id,
            batch_size=batch_size,
            streaming=True,
        )
        return test_gen, asdict(dataset_info)
