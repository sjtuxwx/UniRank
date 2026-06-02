from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from fuxictr.features import FeatureMap


class ModelName(str, Enum):
    DIN = "DIN"
    EST = "EST"
    HEMIX = "HeMix"
    HIFORMER = "HiFormer"
    HYFORMER = "HyFormer"
    INFNET = "INFNet"
    LONGER = "LONGER"
    MIXFORMER = "MixFormer"
    ONETRANS = "OneTrans"
    RANKMIXER = "RankMixer"
    SASREC = "SASRec"
    SASREC_PRETRAIN = "SASRecPretrain"
    SORT = "SORT"
    TOKENFORMER = "TokenFormer"
    TOKENMIXER = "TokenMixer"
    ULTRAHSTU = "UltraHSTU"
    UNIMIXER = "UniMixer"
    ZENITH = "Zenith"


class DatasetName(str, Enum):
    QK = "QK"
    KUAIRAND = "KuaiRand"
    TENCENTGR = "TencentGR"


@dataclass(frozen=True)
class DatasetConfigEntry:
    standard_dataset_id: str
    perday_dataset_id: str | None


@dataclass(frozen=True)
class CheckpointInfo:
    checkpoint: str
    dataset_id: str
    expid: str
    perday: bool
    train_day: str | None = None
    valid_day: str | None = None


class CheckpointLoader:
    @staticmethod
    def normalize_enum_value(value: Enum | str) -> str:
        return value.value if isinstance(value, Enum) else str(value)

    @staticmethod
    def _module_root() -> Path:
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def _load_public_config() -> tuple[dict[str, Any], Path]:
        config_path = CheckpointLoader._module_root() / "config.yaml"
        with config_path.open("r", encoding="utf-8") as cfg:
            return yaml.load(cfg, Loader=yaml.FullLoader), config_path

    @staticmethod
    def _resolve_config_path(path_value: str, config_path: Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (config_path.parent / path).resolve()

    @staticmethod
    def _project_root(config: dict[str, Any], config_path: Path) -> Path:
        project_root = config.get("project_root")
        if not project_root:
            raise ValueError(f"project_root is missing in {config_path}")
        return CheckpointLoader._resolve_config_path(project_root, config_path)

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as cfg:
            return yaml.load(cfg, Loader=yaml.FullLoader)

    @staticmethod
    def _import_model_zoo():
        try:
            import model_zoo
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Failed to import model_zoo while building the checkpoint loader. "
                "Please install the model runtime dependencies first."
            ) from exc
        return model_zoo

    @staticmethod
    def _dataset_entry(dataset_name: DatasetName | str) -> DatasetConfigEntry:
        config, config_path = CheckpointLoader._load_public_config()
        dataset_key = CheckpointLoader.normalize_enum_value(dataset_name)
        datasets = config.get("datasets", {})
        if dataset_key not in datasets:
            available = ", ".join(sorted(datasets.keys()))
            raise ValueError(
                f"Unsupported dataset_name={dataset_key!r}. Available datasets: {available}"
            )
        raw_entry = datasets[dataset_key]
        if "standard_dataset_id" not in raw_entry:
            raise ValueError(
                f"datasets.{dataset_key} must define standard_dataset_id in {config_path}"
            )
        return DatasetConfigEntry(
            standard_dataset_id=raw_entry["standard_dataset_id"],
            perday_dataset_id=raw_entry.get("perday_dataset_id"),
        )

    @staticmethod
    def resolve_dataset_id(dataset_name: DatasetName | str, perday: bool = False) -> str:
        entry = CheckpointLoader._dataset_entry(dataset_name)
        if not perday:
            return entry.standard_dataset_id
        if not entry.perday_dataset_id:
            dataset_key = CheckpointLoader.normalize_enum_value(dataset_name)
            raise ValueError(f"Dataset {dataset_key!r} does not support perday checkpoints.")
        return entry.perday_dataset_id

    @staticmethod
    def resolve_expid(
        model_name: ModelName | str,
        dataset_id: str,
        perday: bool = False,
    ) -> str:
        model_key = CheckpointLoader.normalize_enum_value(model_name)
        suffix = "_rolling" if perday else ""
        return f"{model_key}_{dataset_id}{suffix}"

    @staticmethod
    def _checkpoint_root(config: dict[str, Any], config_path: Path) -> Path:
        checkpoint_root = config.get("checkpoint_root")
        if not checkpoint_root:
            raise ValueError(f"checkpoint_root is missing in {config_path}")
        return CheckpointLoader._resolve_config_path(checkpoint_root, config_path)

    @staticmethod
    def _build_checkpoint_info(
        checkpoint_path: Path,
        dataset_id: str,
        expid: str,
        perday: bool,
    ) -> CheckpointInfo:
        if not perday:
            return CheckpointInfo(
                checkpoint=str(checkpoint_path),
                dataset_id=dataset_id,
                expid=expid,
                perday=False,
            )

        stem = checkpoint_path.stem
        prefix = f"{expid}_"
        suffix = ".model"
        if not stem.startswith(prefix) or "_to_" not in stem:
            raise ValueError(f"Unexpected perday checkpoint name: {checkpoint_path.name}")
        day_part = stem[len(prefix):]
        train_day, valid_day = day_part.split("_to_", 1)
        return CheckpointInfo(
            checkpoint=str(checkpoint_path),
            dataset_id=dataset_id,
            expid=expid,
            perday=True,
            train_day=train_day,
            valid_day=valid_day,
        )

    @staticmethod
    def resolve_checkpoint_path(
        model_name: ModelName | str,
        dataset_name: DatasetName | str,
        perday: bool = False,
        checkpoint_day: str | None = None,
    ) -> Path:
        config, config_path = CheckpointLoader._load_public_config()
        dataset_id = CheckpointLoader.resolve_dataset_id(dataset_name, perday=perday)
        expid = CheckpointLoader.resolve_expid(model_name, dataset_id, perday=perday)
        checkpoint_root = CheckpointLoader._checkpoint_root(config, config_path)
        if not perday:
            checkpoint_path = checkpoint_root / dataset_id / f"{expid}.model"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            return checkpoint_path

        if not checkpoint_day:
            raise ValueError("checkpoint_day is required when perday=True.")
        pattern = f"{expid}_*_to_{checkpoint_day}.model"
        candidates = sorted((checkpoint_root / dataset_id / "rolling").glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No perday checkpoint found for pattern {pattern!r} under "
                f"{checkpoint_root / dataset_id / 'rolling'}"
            )
        if len(candidates) > 1:
            joined = ", ".join(str(path) for path in candidates)
            raise ValueError(
                f"Multiple perday checkpoints match checkpoint_day={checkpoint_day!r}: {joined}"
            )
        return candidates[0]

    @staticmethod
    def list_available_checkpoints(
        model_name: ModelName | str,
        dataset_name: DatasetName | str,
        perday: bool = False,
    ) -> list[CheckpointInfo]:
        config, config_path = CheckpointLoader._load_public_config()
        dataset_id = CheckpointLoader.resolve_dataset_id(dataset_name, perday=perday)
        expid = CheckpointLoader.resolve_expid(model_name, dataset_id, perday=perday)
        checkpoint_root = CheckpointLoader._checkpoint_root(config, config_path)

        if not perday:
            checkpoint_path = checkpoint_root / dataset_id / f"{expid}.model"
            if not checkpoint_path.exists():
                return []
            return [
                CheckpointLoader._build_checkpoint_info(
                    checkpoint_path=checkpoint_path,
                    dataset_id=dataset_id,
                    expid=expid,
                    perday=False,
                )
            ]

        rolling_dir = checkpoint_root / dataset_id / "rolling"
        pattern = f"{expid}_*_to_*.model"
        candidates = sorted(rolling_dir.glob(pattern))
        return [
            CheckpointLoader._build_checkpoint_info(
                checkpoint_path=path,
                dataset_id=dataset_id,
                expid=expid,
                perday=True,
            )
            for path in candidates
        ]

    @staticmethod
    def load_latest_model(
        model_name: ModelName | str,
        dataset_name: DatasetName | str,
        perday: bool = False,
        device: str | int | None = None,
    ):
        checkpoints = CheckpointLoader.list_available_checkpoints(
            model_name=model_name,
            dataset_name=dataset_name,
            perday=perday,
        )
        if not checkpoints:
            dataset_id = CheckpointLoader.resolve_dataset_id(dataset_name, perday=perday)
            expid = CheckpointLoader.resolve_expid(model_name, dataset_id, perday=perday)
            raise FileNotFoundError(
                f"No checkpoints found for expid={expid!r}, dataset_id={dataset_id!r}, "
                f"perday={perday}."
            )

        if not perday:
            return CheckpointLoader.load_model(
                model_name=model_name,
                dataset_name=dataset_name,
                perday=False,
                device=device,
            )

        latest_checkpoint = checkpoints[-1]
        return CheckpointLoader.load_model(
            model_name=model_name,
            dataset_name=dataset_name,
            perday=True,
            checkpoint_day=latest_checkpoint.valid_day,
            device=device,
        )

    @staticmethod
    def _load_runtime_params(expid: str, config: dict[str, Any], config_path: Path) -> dict[str, Any]:
        model_config_path = config.get("model_config")
        dataset_config_path = config.get("dataset_config")
        if not model_config_path or not dataset_config_path:
            raise ValueError(f"model_config and dataset_config must be defined in {config_path}")
        model_config = CheckpointLoader._resolve_config_path(model_config_path, config_path)
        dataset_config = CheckpointLoader._resolve_config_path(dataset_config_path, config_path)
        model_config_dict = CheckpointLoader._read_yaml(model_config)
        dataset_config_dict = CheckpointLoader._read_yaml(dataset_config)

        params = dict(model_config_dict.get("Base", {}))
        if expid not in model_config_dict:
            raise ValueError(f"expid={expid!r} is not defined in {model_config}")
        params.update(model_config_dict[expid])

        dataset_id = params.get("dataset_id")
        if not dataset_id:
            raise ValueError(f"expid={expid!r} does not define dataset_id in {model_config}")
        if dataset_id not in dataset_config_dict:
            raise ValueError(f"dataset_id={dataset_id!r} is not defined in {dataset_config}")

        params.update(dataset_config_dict[dataset_id])
        params["model_id"] = expid
        return params

    @staticmethod
    def _resolve_device(device: str | int | None, config: dict[str, Any]) -> int | str:
        if device is None:
            device = config.get("default_device", "cpu")
        if isinstance(device, int):
            return device
        if isinstance(device, str):
            device_str = device.strip()
            if device_str.lower() == "cpu":
                return -1
            if device_str.isdigit():
                return int(device_str)
        raise ValueError(f"Unsupported device={device!r}. Use 'cpu' or a gpu index like '0'.")

    @staticmethod
    def _feature_map_path(
        params: dict[str, Any],
        project_root: Path,
    ) -> Path:
        data_root = params.get("data_root")
        dataset_id = params["dataset_id"]
        if not data_root:
            raise ValueError("data_root is missing from runtime params.")
        data_root_path = Path(data_root)
        if not data_root_path.is_absolute():
            data_root_path = (project_root / data_root_path).resolve()
        return data_root_path / dataset_id / "feature_map.json"

    @staticmethod
    def build_model(
        model_name: ModelName | str,
        dataset_name: DatasetName | str,
        perday: bool = False,
        device: str | int | None = None,
    ):
        config, config_path = CheckpointLoader._load_public_config()
        project_root = CheckpointLoader._project_root(config, config_path)
        dataset_id = CheckpointLoader.resolve_dataset_id(dataset_name, perday=perday)
        expid = CheckpointLoader.resolve_expid(model_name, dataset_id, perday=perday)
        params = CheckpointLoader._load_runtime_params(expid, config, config_path)
        params["gpu"] = CheckpointLoader._resolve_device(device, config)
        feature_map_path = CheckpointLoader._feature_map_path(params, project_root)
        if not feature_map_path.exists():
            raise FileNotFoundError(
                f"feature_map.json not found for dataset_id={dataset_id}: {feature_map_path}"
            )
        feature_map = FeatureMap(dataset_id, str(feature_map_path.parent))
        feature_map.load(str(feature_map_path), params)
        model_class_name = CheckpointLoader.normalize_enum_value(model_name)
        model_zoo = CheckpointLoader._import_model_zoo()
        if not hasattr(model_zoo, model_class_name):
            raise ValueError(f"model_zoo does not expose model {model_class_name!r}.")
        model_class = getattr(model_zoo, model_class_name)
        model = model_class(feature_map, **params)
        model.model_to_device()
        return model, {
            "dataset_id": dataset_id,
            "expid": expid,
            "feature_map_path": str(feature_map_path),
        }

    @staticmethod
    def load_model(
        model_name: ModelName | str,
        dataset_name: DatasetName | str,
        perday: bool = False,
        checkpoint_day: str | None = None,
        device: str | int | None = None,
    ):
        model_key = CheckpointLoader.normalize_enum_value(model_name)
        dataset_key = CheckpointLoader.normalize_enum_value(dataset_name)
        desc = f"Loading {model_key} on {dataset_key}"
        with tqdm(total=4, desc=desc, unit="step") as pbar:
            pbar.set_postfix_str("resolve checkpoint")
            checkpoint_path = CheckpointLoader.resolve_checkpoint_path(
                model_name=model_name,
                dataset_name=dataset_name,
                perday=perday,
                checkpoint_day=checkpoint_day,
            )
            pbar.update(1)

            pbar.set_postfix_str("build model")
            model, metadata = CheckpointLoader.build_model(
                model_name=model_name,
                dataset_name=dataset_name,
                perday=perday,
                device=device,
            )
            pbar.update(1)

            pbar.set_postfix_str("load weights")
            model.load_weights(str(checkpoint_path))
            pbar.update(1)

            pbar.set_postfix_str("switch eval")
            model.eval()
            pbar.update(1)

        metadata["checkpoint"] = str(checkpoint_path)
        metadata["perday"] = perday
        if checkpoint_day is not None:
            metadata["checkpoint_day"] = checkpoint_day
        return model, metadata
