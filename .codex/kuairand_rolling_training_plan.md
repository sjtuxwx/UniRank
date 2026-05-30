# KuaiRand Per-Day Rolling Training Plan

## 目标

在不修改具体模型代码的前提下，从 `run_expid.py` 增加一套通用 rolling training 逻辑，使 OneTrans、DIN、RankMixer、HyFormer 等所有模型都能复用。

现有训练方式是固定三份数据：

```text
train_data -> valid_data -> test_data
```

目标训练方式是按天滚动：

```text
Day T     训练 K 个 epoch
Day T + 1 验证/测试

Day T + 1 继续训练 K 个 epoch
Day T + 2 验证/测试

...

最后一天作为 final evaluation
```

重点语义：

- 模型只初始化一次。
- 每一天训练结束后，模型参数继续带到下一天。
- 每一天内部训练 `K` 个 epoch。
- 每一天的 validation 使用下一天数据。
- 最后一轮是 `倒数第二天 train -> 最后一天 valid/test`。
- 最后一天的评测结果作为 final AUC。
- 具体模型文件不改，rolling 逻辑只放在训练入口和配置层。

## 当前代码现状

入口文件：

```text
run_expid.py
```

当前主流程大致是：

```python
params = load_config(...)
feature_encoder = FeatureProcessor(**params)
train_data, valid_data, test_data = build_dataset(...)
feature_map = FeatureMap(...)
model = model_class(feature_map, **params)

train_gen, valid_gen = RankDataLoader(..., stage="train", **params).make_iterator()
model.fit(train_gen, validation_data=valid_gen, **params)

test_gen = RankDataLoader(..., stage="test", **params).make_iterator()
model.evaluate(test_gen)
```

这个流程默认只有一次训练、一次验证、一次测试。

模型侧现状：

- 各模型只负责定义结构和 `forward()`。
- 训练统一走 `model.fit(...)`。
- 评测统一走 `model.evaluate(...)`。
- 因此 rolling 不应该放进 `model_zoo/*.py`。

数据侧现状：

```text
data/KuaiRand_Video_Action_perday/
  0407.parquet
  0408.parquet
  ...
  0508.parquet
  user_info.parquet
  item_info.parquet
  meta_data.json
```

这些每日 parquet schema 应该保持一致，可以复用同一份：

```text
feature_map.json
user_info.parquet
item_info.parquet
meta_data.json
```

## 推荐配置设计

在 `config/dataset_config.yaml` 中保留 per-day 数据集配置，例如：

```yaml
KuaiRand_Video_Action_perday:
    data_root: ./data/
    data_format: parquet
    rolling_data_dir: ./data/KuaiRand_Video_Action_perday
    user_info: ./data/KuaiRand_Video_Action_perday/user_info.parquet
    item_info: ./data/KuaiRand_Video_Action_perday/item_info.parquet
    rebuild_dataset: False
    feature_cols:
        ...
    label_col:
        ...
```

为了兼容旧逻辑，也可以继续保留：

```yaml
train_data: ./data/KuaiRand_Video_Action_perday/0407.parquet
valid_data: ./data/KuaiRand_Video_Action_perday/0408.parquet
test_data: ./data/KuaiRand_Video_Action_perday/0408.parquet
```

但 rolling 模式下优先使用 `rolling_data_dir` 扫描每日文件。

在 `config/model_config.yaml` 中新增 rolling 实验配置：

```yaml
OneTrans_KuaiRand_Video_Action_perday_rolling:
    model: OneTrans
    dataset_id: KuaiRand_Video_Action_perday
    rolling_train: True
    rolling_epochs_per_day: 3
    rolling_start_day: "0407"
    rolling_end_day: "0508"
    rolling_final_metric: AUC
    ...
```

其中：

- `rolling_train`: 是否启用 rolling 逻辑。
- `rolling_epochs_per_day`: 每一天内部训练多少个 epoch。
- `rolling_start_day`: 起始训练日，可选。
- `rolling_end_day`: 最终评测日，可选。
- `rolling_final_metric`: 最终关注指标，默认沿用 `monitor`。

## run_expid.py 改造计划

### 1. 保留现有标准训练路径

当前逻辑不要删除，只封装成类似：

```python
def run_standard_training(model, feature_map, params, distributed, rank, local_rank, world_size):
    ...
```

原有非 rolling 实验仍然走它。

判断入口：

```python
if params.get("rolling_train", False):
    run_rolling_training(...)
else:
    run_standard_training(...)
```

这样可以保证所有旧实验不受影响。

### 2. 新增每日文件扫描函数

新增 helper：

```python
def list_rolling_day_files(rolling_data_dir, start_day=None, end_day=None):
    ...
```

行为：

- 扫描 `*.parquet`。
- 只保留文件名形如 `MMDD.parquet` 的文件。
- 排除 `user_info.parquet`、`item_info.parquet` 等 side info。
- 按日期字符串排序。
- 如果配置了 `start_day` / `end_day`，做闭区间过滤。

输出：

```python
[
    ("0407", "./data/.../0407.parquet"),
    ("0408", "./data/.../0408.parquet"),
    ...
]
```

### 3. 新增 rolling pair 构造函数

新增 helper：

```python
def build_rolling_pairs(day_files):
    ...
```

输出：

```python
[
    {
        "train_day": "0407",
        "train_data": ".../0407.parquet",
        "valid_day": "0408",
        "valid_data": ".../0408.parquet",
        "is_final": False,
    },
    ...
    {
        "train_day": "0507",
        "train_data": ".../0507.parquet",
        "valid_day": "0508",
        "valid_data": ".../0508.parquet",
        "is_final": True,
    },
]
```

如果有效天数少于 2 天，直接报错。

### 4. 新增 dataloader 构造函数

为了减少重复代码，封装：

```python
def make_train_valid_loaders(feature_map, params, train_data, valid_data, distributed, rank, local_rank, world_size):
    ...
```

内部设置：

```python
day_params = dict(params)
day_params["train_data"] = train_data
day_params["valid_data"] = valid_data
day_params["data_loader"] = UniRankDataloader
```

然后调用：

```python
RankDataLoader(feature_map, stage="train", **day_params).make_iterator()
```

这样 rolling 每一天都可以动态换数据文件。

### 5. 新增 rolling training 主循环

核心逻辑：

```python
def run_rolling_training(model, feature_map, params, distributed, rank, local_rank, world_size):
    day_files = list_rolling_day_files(...)
    rolling_pairs = build_rolling_pairs(day_files)

    rolling_epochs = int(params.get("rolling_epochs_per_day", params.get("epochs", 1)))

    for pair in rolling_pairs:
        day_params = dict(params)
        day_params["epochs"] = rolling_epochs
        day_params["train_data"] = pair["train_data"]
        day_params["valid_data"] = pair["valid_data"]

        train_gen, valid_gen = make_train_valid_loaders(...)
        model.fit(train_gen, validation_data=valid_gen, **day_params)

        cleanup loaders

    final_pair = rolling_pairs[-1]
    run final evaluate on final_pair["valid_data"]
```

注意：

- 模型对象不重新创建。
- optimizer 不重新创建。
- `model.fit()` 每天会重置 best metric / early stop 状态，但不会重置模型参数。
- 因为 `fit()` 结束时会 `load_weights(self.checkpoint)`，所以每天结束后模型会回到当天 valid 上最优的 epoch。

这刚好符合“每天内部 K 个 epoch，保留当天 valid AUC 最优模型，然后继续下一天”的语义。

### 6. checkpoint 设计

当前 `model.fit()` 使用 `self.checkpoint` 保存 best model。

Rolling 下有两个可选设计：

方案 A：所有天共用一个 checkpoint。

优点：

- 改动最小。
- 最后一轮 best model 会留在默认 checkpoint。

缺点：

- 中间每天的 best model 会被覆盖。
- 不容易回溯每天结果。

方案 B：每个 rolling step 使用独立 checkpoint。

示例：

```text
checkpoints/KuaiRand_Video_Action_perday/OneTrans_.../rolling_0407_to_0408.model
checkpoints/KuaiRand_Video_Action_perday/OneTrans_.../rolling_0408_to_0409.model
...
```

实现方式：

```python
base_checkpoint = model.checkpoint
model.checkpoint = make_rolling_checkpoint(base_checkpoint, train_day, valid_day)
model.fit(...)
```

推荐第一版采用方案 B，因为日志和模型文件更清楚。最后一轮 `0507_to_0508` 的 checkpoint 就是 final checkpoint。

### 7. final evaluation 设计

最后一轮训练已经用最后一天作为 validation：

```text
train = 0507
valid = 0508
```

`fit()` 会根据 `0508` validation AUC 保存 best model，并在结束时加载它。

然后再执行：

```python
test_gen = RankDataLoader(..., stage="test", test_data="0508.parquet")
model.evaluate(test_gen)
```

这个结果作为 final evaluation。

注意：这会对 `0508` 评估两次。

- 第一次在最后一轮每个 epoch 后作为 validation。
- 第二次训练结束后作为 final test。

这和你的描述一致：最后一天就是最终评测数据，同时用于最后一轮选 best epoch。

## DDP 兼容性计划

Rolling 本身可以兼容 DDP，但第一版需要小心同步点。

每个 rolling step：

1. 所有 rank 使用同一组 `train_data` / `valid_data`。
2. 构建 dataloader 前可以 `dist.barrier()`。
3. `model.fit()` 内部已有 DDP blocked dataloader 的同步逻辑。
4. 每个 step 结束后再 `dist.barrier()`。
5. 切换下一天前清理 dataloader 和 GPU cache。

如果第一版只要求单卡，可以先明确：

```python
if rolling_train and distributed:
    raise NotImplementedError("Rolling training with DDP is not enabled yet.")
```

但从长期看，建议让它支持 DDP，因为 `run_expid.py` 已经有比较完整的 DDP 框架。

## 日志与结果记录

建议新增一个 rolling summary 文件，例如：

```text
checkpoints/<dataset_id>/<model_id>/rolling_results.json
```

记录内容：

```json
{
  "rolling_pairs": [
    {
      "train_day": "0407",
      "valid_day": "0408",
      "train_data": ".../0407.parquet",
      "valid_data": ".../0408.parquet",
      "checkpoint": ".../rolling_0407_to_0408.model"
    }
  ],
  "final_test_day": "0508",
  "final_checkpoint": ".../rolling_0507_to_0508.model"
}
```

第一版可以先只写日志，不写 JSON。第二版再加结果文件。

## 配置示例

新增 dataset：

```yaml
KuaiRand_Video_Action_perday:
    data_root: ./data/
    data_format: parquet
    rolling_data_dir: ./data/KuaiRand_Video_Action_perday
    train_data: ./data/KuaiRand_Video_Action_perday/0407.parquet
    valid_data: ./data/KuaiRand_Video_Action_perday/0408.parquet
    test_data: ./data/KuaiRand_Video_Action_perday/0408.parquet
    user_info: ./data/KuaiRand_Video_Action_perday/user_info.parquet
    item_info: ./data/KuaiRand_Video_Action_perday/item_info.parquet
    rebuild_dataset: False
    feature_cols:
        ...
    label_col:
        ...
```

新增 model expid：

```yaml
OneTrans_KuaiRand_Video_Action_perday_rolling:
    model: OneTrans
    dataset_id: KuaiRand_Video_Action_perday
    rolling_train: True
    rolling_epochs_per_day: 3
    rolling_start_day: "0407"
    rolling_end_day: "0508"
    loss: [...]
    metrics: ['logloss', 'AUC', 'gAUC']
    ...
```

运行命令：

```bash
python run_expid.py \
  --config ./config \
  --expid OneTrans_KuaiRand_Video_Action_perday_rolling \
  --gpu 0
```

同样的 rolling 配置可以复用于其他模型：

```yaml
DIN_KuaiRand_Video_Action_perday_rolling:
    model: DIN
    dataset_id: KuaiRand_Video_Action_perday
    rolling_train: True
    rolling_epochs_per_day: 3
    ...
```

## 验证计划

第一步：配置解析验证。

确认：

- `rolling_train=True` 能被 `load_config()` 读到。
- `rolling_data_dir` 存在。
- 每日 parquet 至少有 2 个。
- `feature_map.json` 能正常加载。

第二步：两天小跑。

只跑：

```text
0407 -> 0408
```

配置：

```yaml
rolling_start_day: "0407"
rolling_end_day: "0408"
rolling_epochs_per_day: 1
```

目标：

- dataloader 能构建。
- model.fit 能跑完。
- final evaluate 能跑完。

第三步：三天连续训练。

跑：

```text
0407 -> 0408
0408 -> 0409
```

目标：

- 第二天训练没有重新初始化模型。
- checkpoint 切换正常。
- 日志能看出 step 顺序。

第四步：完整 rolling。

跑：

```text
0407 -> 0508
```

目标：

- 最后一轮 `0507 -> 0508` 能保留 best checkpoint。
- final AUC 输出明确。

## 风险点

### 1. `model.fit()` 每天都会 load best checkpoint

这是符合当前需求的，但要明确：

```text
每天 K epoch 后，继续下一天训练的是“当天 valid 上最优”的模型，而不是当天最后一个 epoch 的模型。
```

如果你希望继续训练最后一个 epoch，而只在最终一天选 best，那要改策略。

### 2. optimizer 状态是否跨天延续

默认模型不重建，optimizer 也不重建，所以 Adam 动量会跨天保留。

这通常符合“继续训练”。

如果你希望每天重置 optimizer，需要额外参数：

```yaml
rolling_reset_optimizer_each_day: True
```

第一版建议不重置。

### 3. checkpoint 文件覆盖

如果不改 checkpoint 名，中间天的 best 会被覆盖。

推荐 rolling 模式下每个 pair 单独 checkpoint。

### 4. final day 同时作为 validation 和 final evaluation

当前计划会这样做：

```text
0508 用于 0507 训练时的 epoch selection
0508 也用于最后 final evaluation
```

这和你的描述一致，但严格机器学习评测里它不是完全独立 test。

如果你需要纯 final test，不参与选 epoch，就要另外留出一天或改变选择策略。

### 5. DDP first version scope

DDP 可以支持，但要认真测同步。

如果时间紧，第一版建议先单卡跑通。

## 待确认问题

1. 每天训练 K 个 epoch 后，下一天继续训练的模型，是“当天 valid 最优 epoch”的模型，还是“当天最后一个 epoch”的模型？

当前计划：使用“当天 valid 最优 epoch”的模型。

2. Optimizer 状态是否跨天保留？

当前计划：保留。

3. 最后一天是否允许参与最后一轮 epoch selection？

当前计划：允许。也就是 `0507` 训练时用 `0508` 选 best，之后再在 `0508` 上 final evaluate。

4. 是否需要支持 DDP rolling？

当前计划：设计上兼容，但建议第一轮先单卡验证。

5. 是否需要保存每天的 best checkpoint？

当前计划：保存每个 pair 的 best checkpoint，最后一轮 checkpoint 是 final checkpoint。

6. rolling 文件是否永远是 `MMDD.parquet` 命名？

当前计划：只扫描 `^\d{4}\.parquet$`，例如 `0407.parquet`。

## 推荐实施顺序

1. 修改 `run_expid.py`，把标准训练流程封装成函数。
2. 新增 rolling 文件扫描和 pair 构造 helper。
3. 新增 `run_rolling_training()`。
4. 新增 rolling config。
5. 用 `0407 -> 0408` 单卡跑通。
6. 用 `0407 -> 0409` 验证连续训练。
7. 跑完整 `0407 -> 0508`。

