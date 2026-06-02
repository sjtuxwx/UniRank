from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from fair_tool.evaluator.dataset_reader import EvaluationDatasetReader
from fair_tool.utils.checkpoint_loader import CheckpointLoader, DatasetName


class RecommendationResultGenerator:
    @staticmethod
    def _load_public_config() -> tuple[dict[str, Any], Path]:
        return CheckpointLoader._load_public_config()

    @staticmethod
    def _resolve_output_dir(
        dataset_name: DatasetName | str,
        eval_date: str,
        model,
        output_root: str | None = None,
    ) -> Path:
        config, config_path = RecommendationResultGenerator._load_public_config()
        dataset_key = CheckpointLoader.normalize_enum_value(dataset_name)
        model_name = model.__class__.__name__
        if output_root is None:
            output_root = config.get("result_root", "./results")
        output_root_path = CheckpointLoader._resolve_config_path(output_root, config_path)
        output_dir = output_root_path / f"{dataset_key}_{eval_date}_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _resolve_metrics(model, metrics: list[str] | None) -> list[str]:
        if metrics is not None:
            return metrics
        if hasattr(model, "validation_metrics") and model.validation_metrics:
            return list(model.validation_metrics)
        return []

    @staticmethod
    def _resolve_labels(model) -> list[str]:
        labels = list(getattr(model.feature_map, "labels", []))
        if not labels:
            raise ValueError("model.feature_map.labels is empty; cannot generate task results.")
        return labels

    @staticmethod
    def _extract_user_ids(batch_data) -> np.ndarray:
        batch_dict = batch_data[0]
        if "user_index" not in batch_dict:
            raise KeyError("batch_data does not contain 'user_index'.")
        return batch_dict["user_index"].detach().cpu().numpy().reshape(-1)

    @staticmethod
    def _extract_group_ids(model, batch_data) -> np.ndarray | None:
        if getattr(model.feature_map, "group_id", None) is None:
            return None
        return model.get_group_id(batch_data).detach().cpu().numpy().reshape(-1)

    @staticmethod
    def _extract_predictions(model, batch_data, labels: list[str]) -> dict[str, np.ndarray]:
        return_dict = model.forward(batch_data)
        predictions = {}
        for label in labels:
            pred_key = f"{label}_pred"
            if pred_key in return_dict:
                predictions[label] = return_dict[pred_key].detach().cpu().numpy().reshape(-1)
            elif "y_pred" in return_dict and len(labels) == 1:
                predictions[label] = return_dict["y_pred"].detach().cpu().numpy().reshape(-1)
            else:
                raise KeyError(f"Prediction key {pred_key!r} is missing from model output.")
        return predictions

    @staticmethod
    def _extract_labels(model, batch_data, labels: list[str]) -> dict[str, np.ndarray]:
        y_true = model.get_labels(batch_data)
        if isinstance(y_true, (list, tuple)):
            if len(y_true) != len(labels):
                raise ValueError(
                    f"Number of labels from model.get_labels()={len(y_true)} does not match "
                    f"feature_map.labels={len(labels)}."
                )
            return {
                label: y_true[idx].detach().cpu().numpy().reshape(-1)
                for idx, label in enumerate(labels)
            }
        if len(labels) != 1:
            raise ValueError("Single tensor labels are only supported for single-task models.")
        return {labels[0]: y_true.detach().cpu().numpy().reshape(-1)}

    @staticmethod
    def _task_file_path(output_dir: Path, task_name: str) -> Path:
        return output_dir / f"{task_name}.csv"

    @staticmethod
    def _tmp_task_file_path(output_dir: Path, task_name: str) -> Path:
        tmp_dir = output_dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir / f"{task_name}_raw.csv"

    @staticmethod
    def _needs_both_classes(metrics: list[str]) -> bool:
        return any(metric in {"AUC", "gAUC", "avgAUC"} for metric in metrics)

    @staticmethod
    def _has_both_classes(y_true: np.ndarray) -> bool:
        yb = (np.asarray(y_true).reshape(-1) > 0).astype(np.int8)
        pos = int(yb.sum())
        neg = len(yb) - pos
        return pos > 0 and neg > 0

    @staticmethod
    def _compute_user_metrics(
        model,
        metrics: list[str],
        user_id: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any] | None:
        if RecommendationResultGenerator._needs_both_classes(metrics):
            if not RecommendationResultGenerator._has_both_classes(y_true):
                return None

        row = {"user_index": int(user_id), "sample_count": int(len(y_true))}
        metric_errors = {}
        for metric in metrics:
            try:
                if metric in {"gAUC", "avgAUC", "MRR"} or metric.startswith("NDCG"):
                    group_id = np.full(len(y_true), int(user_id), dtype=np.int64)
                    result = model.evaluate_metrics(y_true, y_pred, [metric], group_id)
                else:
                    result = model.evaluate_metrics(y_true, y_pred, [metric], None)
                row[metric] = result.get(metric)
            except Exception as exc:
                row[metric] = None
                metric_errors[metric] = str(exc)

        if metric_errors:
            row["_errors"] = metric_errors
        return row

    @staticmethod
    def _write_user_metric_rows(task_file: Path, metric_rows: list[dict[str, Any]], metrics: list[str]) -> int:
        fieldnames = ["user_index", "sample_count"] + list(metrics)
        with task_file.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in metric_rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        return len(metric_rows)

    @staticmethod
    def _append_raw_rows(task_file: Path, user_ids, labels, preds) -> int:
        write_header = not task_file.exists()
        with task_file.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["user_index", "label", "pred"])
            for user_id, label, pred in zip(user_ids, labels, preds):
                writer.writerow([int(user_id), float(label), float(pred)])
        return len(preds)

    @staticmethod
    def _build_task_user_rows(model, metrics, user_ids, labels, preds) -> list[dict[str, Any]]:
        grouped = defaultdict(lambda: {"y_true": [], "y_pred": []})
        for user_id, label, pred in zip(user_ids, labels, preds):
            bucket = grouped[int(user_id)]
            bucket["y_true"].append(label)
            bucket["y_pred"].append(pred)

        rows = []
        for user_id in sorted(grouped.keys()):
            y_true = np.array(grouped[user_id]["y_true"], dtype=np.float64)
            y_pred = np.array(grouped[user_id]["y_pred"], dtype=np.float64)
            metric_row = RecommendationResultGenerator._compute_user_metrics(
                model=model,
                metrics=metrics,
                user_id=user_id,
                y_true=y_true,
                y_pred=y_pred,
            )
            if metric_row is not None:
                rows.append(metric_row)
        return rows

    @staticmethod
    def _build_task_user_rows_from_raw_file(model, metrics, raw_task_file: Path) -> list[dict[str, Any]]:
        grouped = defaultdict(lambda: {"y_true": [], "y_pred": []})
        with raw_task_file.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user_id = int(row["user_index"])
                grouped[user_id]["y_true"].append(float(row["label"]))
                grouped[user_id]["y_pred"].append(float(row["pred"]))

        rows = []
        for user_id in sorted(grouped.keys()):
            y_true = np.array(grouped[user_id]["y_true"], dtype=np.float64)
            y_pred = np.array(grouped[user_id]["y_pred"], dtype=np.float64)
            metric_row = RecommendationResultGenerator._compute_user_metrics(
                model=model,
                metrics=metrics,
                user_id=user_id,
                y_true=y_true,
                y_pred=y_pred,
            )
            if metric_row is not None:
                rows.append(metric_row)
        return rows

    @staticmethod
    def _compute_summary(model, labels, metrics, y_true_all, y_pred_all, group_ids):
        summary = {"tasks": {}}
        for label in labels:
            y_true = np.array(y_true_all[label], dtype=np.float64)
            y_pred = np.array(y_pred_all[label], dtype=np.float64)
            task_metrics = {}
            metric_errors = {}
            for metric in metrics:
                try:
                    result = model.evaluate_metrics(y_true, y_pred, [metric], group_ids)
                    task_metrics[metric] = result.get(metric)
                except Exception as exc:
                    task_metrics[metric] = None
                    metric_errors[metric] = str(exc)
            if metric_errors:
                task_metrics["_errors"] = metric_errors
            summary["tasks"][label] = dict(task_metrics)
        return summary

    @staticmethod
    def _write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=True)

    @staticmethod
    def generate_results(
        model,
        dataset_name: DatasetName | str,
        eval_date: str,
        use_streaming: bool = False,
        batch_size: int | None = None,
        output_root: str | None = None,
        metrics: list[str] | None = None,
    ):
        if not hasattr(model, "feature_map"):
            raise ValueError("model must expose feature_map.")
        if not hasattr(model, "model_id"):
            raise ValueError("model must expose model_id.")

        reader_fn = (
            EvaluationDatasetReader.stream_batches if use_streaming
            else EvaluationDatasetReader.load_all
        )
        test_gen, dataset_info = reader_fn(
            dataset_name=dataset_name,
            eval_date=eval_date,
            feature_map=model.feature_map,
            model_id=model.model_id,
            batch_size=batch_size,
        )

        output_dir = RecommendationResultGenerator._resolve_output_dir(
            dataset_name=dataset_name,
            eval_date=eval_date,
            model=model,
            output_root=output_root,
        )
        summary_path = output_dir / "summary.json"
        labels = RecommendationResultGenerator._resolve_labels(model)
        resolved_metrics = RecommendationResultGenerator._resolve_metrics(model, metrics)

        model.eval()
        y_true_all = defaultdict(list)
        y_pred_all = defaultdict(list)
        user_ids_all = defaultdict(list)
        group_ids = []
        task_files = {
            label: RecommendationResultGenerator._task_file_path(output_dir, label)
            for label in labels
        }
        tmp_task_files = {
            label: RecommendationResultGenerator._tmp_task_file_path(output_dir, label)
            for label in labels
        } if use_streaming else {}

        with torch.no_grad():
            batch_iterator = test_gen
            try:
                total_batches = len(test_gen)
            except TypeError:
                total_batches = None
            batch_iterator = tqdm(
                batch_iterator,
                total=total_batches,
                desc=f"Evaluating {CheckpointLoader.normalize_enum_value(dataset_name)} {eval_date}",
                unit="batch",
                file=sys.stdout,
            )
            for batch_data in batch_iterator:
                batch_user_ids = RecommendationResultGenerator._extract_user_ids(batch_data)
                batch_group_ids = RecommendationResultGenerator._extract_group_ids(model, batch_data)
                batch_preds = RecommendationResultGenerator._extract_predictions(model, batch_data, labels)
                batch_labels = RecommendationResultGenerator._extract_labels(model, batch_data, labels)

                if batch_group_ids is not None:
                    group_ids.extend(batch_group_ids.tolist())

                for label in labels:
                    y_true_all[label].extend(batch_labels[label].tolist())
                    y_pred_all[label].extend(batch_preds[label].tolist())
                    if use_streaming:
                        RecommendationResultGenerator._append_raw_rows(
                            task_file=tmp_task_files[label],
                            user_ids=batch_user_ids,
                            labels=batch_labels[label],
                            preds=batch_preds[label],
                        )
                    else:
                        user_ids_all[label].extend(batch_user_ids.tolist())

        task_row_counts = {}
        task_iterator = tqdm(labels, desc="Writing task files", unit="task", file=sys.stdout)
        for label in task_iterator:
            if use_streaming:
                metric_rows = RecommendationResultGenerator._build_task_user_rows_from_raw_file(
                    model=model,
                    metrics=resolved_metrics,
                    raw_task_file=tmp_task_files[label],
                )
            else:
                metric_rows = RecommendationResultGenerator._build_task_user_rows(
                    model=model,
                    metrics=resolved_metrics,
                    user_ids=user_ids_all[label],
                    labels=y_true_all[label],
                    preds=y_pred_all[label],
                )
            task_row_counts[label] = RecommendationResultGenerator._write_user_metric_rows(
                task_file=task_files[label],
                metric_rows=metric_rows,
                metrics=resolved_metrics,
            )

        summary = RecommendationResultGenerator._compute_summary(
            model=model,
            labels=labels,
            metrics=resolved_metrics,
            y_true_all=y_true_all,
            y_pred_all=y_pred_all,
            group_ids=np.array(group_ids) if group_ids else None,
        )
        summary.update({
            "dataset_name": CheckpointLoader.normalize_enum_value(dataset_name),
            "dataset_id": dataset_info["dataset_id"],
            "eval_date": str(eval_date),
            "model_name": model.__class__.__name__,
            "model_id": model.model_id,
            "use_streaming": use_streaming,
            "metrics": resolved_metrics,
            "output_dir": str(output_dir),
            "task_files": {label: str(task_files[label]) for label in labels},
            "num_rows": task_row_counts,
        })
        RecommendationResultGenerator._write_summary(summary_path, summary)
        if use_streaming:
            for path in tmp_task_files.values():
                if path.exists():
                    path.unlink()
            tmp_dir = output_dir / ".tmp"
            if tmp_dir.exists():
                try:
                    tmp_dir.rmdir()
                except OSError:
                    pass

        return {
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "task_files": {label: str(task_files[label]) for label in labels},
            "dataset_info": dataset_info,
            "metrics": resolved_metrics,
            "num_rows": task_row_counts,
        }
