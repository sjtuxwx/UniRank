from __future__ import annotations

from fair_tool.evaluator import RecommendationResultGenerator
from fair_tool.utils import CheckpointLoader, DatasetName, ModelName


def generate_recommendation_results(
    model_name: ModelName | str,
    checkpoint_time: str,
    dataset_name: DatasetName | str,
    metrics: list[str],
    test_time: str | None = None,
    use_streaming: bool = True,
    device: str | int | None = None,
    output_root: str | None = None,
    batch_size: int | None = 256,
):
    """加载已保存的模型 checkpoint，执行评测，并导出结果文件。"""
    checkpoint_time = str(checkpoint_time)
    # 如果没有显式指定 test_time，就默认在与 checkpoint 相同的切分/日期上评测。
    test_time = checkpoint_time if test_time is None else str(test_time)
    # 类似 "0508" 这样的四位字符串，会被视为按天存储的 checkpoint / 数据集。
    perday = len(checkpoint_time) == 4 and checkpoint_time.isdigit()

    # 先根据模型名、数据集名和日期定位 checkpoint，并恢复训练好的模型。
    model, model_info = CheckpointLoader.load_model(
        model_name=model_name,
        dataset_name=dataset_name,
        perday=perday,
        checkpoint_day=checkpoint_time if perday else None,
        device=device,
    )

    # 再运行评测流程，并把每个任务的 CSV 结果以及 summary.json 写到输出目录。
    result = RecommendationResultGenerator.generate_results(
        model=model,
        dataset_name=dataset_name,
        eval_date=test_time,
        use_streaming=use_streaming,
        batch_size=batch_size,
        output_root=output_root,
        metrics=metrics,
    )
    # 额外补充一些运行时信息，方便调用方追踪这份结果对应的 checkpoint 和评测设置。
    result["model_info"] = model_info
    result["checkpoint_time"] = checkpoint_time
    result["test_time"] = test_time
    result["use_streaming"] = use_streaming
    return result
