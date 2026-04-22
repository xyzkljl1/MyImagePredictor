# ImagePopularity

> 声明：
> 本项目完全通过 vibe coding 方式生成。
> 主要生成代理为 OpenAI Codex（GPT-5 系列编码代理），基于用户给出的需求持续协作完成设计、实现、调试与文档整理。

基于 `TorchSharp + CUDA` 的图片偏好二分类方案，包含：

- `ImagePopularity.Core`：可复用类库，提供单图与批量概率预测接口。
- `ImagePopularity.Trainer`：训练入口（两个目录：正样本目录/负样本目录）。
- `ImagePopularity.Demo`：目录批量推理示例。

## 1. 数据准备

把 1 万张已标注图片分到两个目录：

- `D:\data\popular`（目录下全部是受欢迎图片，label=1）
- `D:\data\unpopular`（目录下全部是不受欢迎图片，label=0）

候选 70 万张图片可以放在：

- `D:\data\candidates`

支持扩展名：`.jpg .jpeg .png .bmp .gif .webp .tif .tiff`

格式处理说明：

- `.jpg` / `.jpeg` / `.png` / `.bmp` / `.webp` / `.tif` / `.tiff`：按普通静态图处理，解码后统一转换到 `RGB`，再执行 `AutoOrient`、可选数据增强、`PadResize` 和 `Normalize`。
- `.gif`：仅抽取首帧参与训练和预测，不会利用后续动画帧信息；后续流程与静态图相同。
- 无论原始格式是什么，进入模型前都会被转换成同一种 `float32 CHW` 张量表示，所以模型本身看不到“文件格式”这个概念，只看到预处理后的像素张量。
- 当前代码中，图片格式白名单以 [SupportedImageFiles.cs](/E:/MyWebsiteHelper/codex/src/ImagePopularity.Core/SupportedImageFiles.cs) 为准；如果后续新增、删除或调整某种图片格式的处理方式，README 这里也应同步更新。

## 2. 训练模型

最小调用命令示例：

```powershell
dotnet run --project src/ImagePopularity.Trainer -- \
  --popular-dir D:\data\popular \
  --unpopular-dir D:\data\unpopular
```

使用显式验证集子目录的最小调用命令示例：

```powershell
dotnet run --project src/ImagePopularity.Trainer -- \
  --popular-dir D:\data\popular \
  --unpopular-dir D:\data\unpopular \
  --validation-dir validation
```

使用多个显式验证集子目录的示例：

```powershell
dotnet run --project src/ImagePopularity.Trainer -- \
  --popular-dir D:\data\popular \
  --unpopular-dir D:\data\unpopular \
  --validation-dir validation \
  --validation-dir holdout
```

全参数调用命令示例：

```powershell
dotnet run --project src/ImagePopularity.Trainer -- \
  --popular-dir D:\data\popular \
  --unpopular-dir D:\data\unpopular \
  --validation-dir validation \
  --output-model all_ \
  --preprocess-cache-dir models\preprocess-cache \
  --backbone resnet152 \
  --freeze-backbone-epochs 3 \
  --epochs 30 \
  --batch-size 256 \
  --train-image-size 320 \
  --enable-augmentation true \
  --hflip-prob 0.5 \
  --max-rotation-deg 12 \
  --brightness-jitter 0.15 \
  --contrast-jitter 0.15 \
  --saturation-jitter 0.15 \
  --min-random-crop-scale 0.85 \
  --learning-rate 0.0003 \
  --fine-tune-learning-rate 0.00005 \
  --weight-decay 0.0001 \
  --validation-split 0.1
```

训练会输出：

- 模型权重：自动命名，例如 `models/all_9000_320_a1_e30b256s42_04212210.pt`
- 元信息：与模型同名的 `.meta.json` 文件

`--output-model` 现在只表示“自动命名模型文件名前缀”，不是路径，也不会改变模型输出目录。
例如传入 `all_` 后，程序会在 `models` 目录下生成以 `all_` 开头的自动命名模型文件。

验证集选择说明：

- 不传 `--validation-dir` 时：程序会继续使用原来的随机分层切分，按 `--validation-split` 从正负两类样本中各自抽取一部分作为验证集。
- 传入 `--validation-dir` 时：它表示相对于 `--popular-dir` 和 `--unpopular-dir` 的一个或多个同名验证子目录；可以重复传入多个 `--validation-dir`，也可以在一个参数里用 `,` 或 `;` 分隔多个目录名。
- 例如传入 `--validation-dir validation` 后，程序会把：
  - `D:\data\popular\validation\**`
  - `D:\data\unpopular\validation\**`
  中的图片用作验证集。
- 如果同时传入多个验证子目录，程序会把所有**实际存在**的验证子目录里的图片合并起来作为验证集。
- 这些验证子目录中的图片会自动从训练集中排除，所以即使验证集目录嵌在训练目录内部，也不会被同时用于训练。
- 如果其中一个或多个验证子目录不存在，程序会直接忽略这些不存在的目录，不会报错，也不会回退到随机切分。
- 程序最终只按“所有实际存在的验证子目录中的图片总数”判断是否足够；如果这个总数小于 `总图片数 × validation-split`，程序会直接报错并停止训练。

预训练骨干微调说明：

- 预训练骨干已强制开启，会自动下载与 `--backbone` 对应的 TorchSharp 预训练权重（缓存到 `models/pretrained`）。
- 前 `--freeze-backbone-epochs` 个 epoch 冻结骨干网络，只训练分类头，先稳定收敛。
- 之后自动解冻骨干，并切到 `--fine-tune-learning-rate` 做全网微调。
- 当前默认骨干为 `resnet152`。
- 训练预处理缓存始终启用，默认目录 `models/preprocess-cache`，可通过 `--preprocess-cache-dir` 指定。
- 训练缓存内容为“增强 + PadResize + Normalize”后的 `float32 CHW` 张量（`train` 子目录）；验证缓存为“PadResize + Normalize”后的张量（`validation` 子目录）。
- `Trainer` 不再提供 `--inference-image-size` 参数，模型元数据中的“推荐推理尺寸”会自动等于 `--train-image-size`；推理时你仍可在 `Demo/API` 里传入 `inferenceImageSize` 覆盖。

如果你已经有本地预训练权重文件，可指定：

```powershell
--pretrained-weights D:\weights\ResNet50_Weights.IMAGENET1K_V2
```

## 3. 类库推理 API

单图预测：

```csharp
using ImagePopularity.Core;

using var predictor = new ImagePopularityPredictor(
    modelPath: @"models\all_9000_320_a1_e30b256s42_04212210.pt",
    options: new ImagePopularityPredictorOptions
    {
        InferenceImageSize = 384, // 可与训练分辨率不同
        EnablePreprocessCache = false // 默认 false，推理缓存按需开启
    });

float probability = predictor.PredictProbability(@"D:\data\candidates\sample.jpg");
Console.WriteLine($"Popular probability = {probability:P2}");
```

批量预测（推荐）：

```csharp
using ImagePopularity.Core;

var imagePaths = Directory.EnumerateFiles(@"D:\data\candidates", "*", SearchOption.AllDirectories)
    .Where(path => new[] { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff" }
        .Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
    .OrderBy(path => path)
    .ToList();

using var predictor = new ImagePopularityPredictor(@"models\all_9000_320_a1_e30b256s42_04212210.pt",
    new ImagePopularityPredictorOptions
    {
        InferenceImageSize = 384,
        EnablePreprocessCache = true,
        PreprocessCacheDirectory = @"models\inference-cache"
    });

var probabilities = predictor.PredictProbabilities(imagePaths, batchSize: 256);
for (var i = 0; i < imagePaths.Count; i++)
{
    Console.WriteLine($"{Path.GetFileName(imagePaths[i])}\t{probabilities[i]:F6}");
}
```

## 4. Demo 目录批量推理

最小调用命令示例：

```powershell
dotnet run --project src/ImagePopularity.Demo -- \
  models\all_9000_320_a1_e30b256s42_04212210.pt \
  D:\data\candidates
```

全参数调用命令示例：

```powershell
dotnet run --project src/ImagePopularity.Demo -- \
  models\all_9000_320_a1_e30b256s42_04212210.pt \
  D:\data\candidates \
  256 \
  384 \
  true \
  models\inference-cache
```

输出格式：

- 只输出 `预测概率 > 0.5` 的图片：`文件名<TAB>预测概率`
- 最后额外输出：
  - 所有图片的平均概率
  - `> 0.5` 的图片数量
  - 图片总数

参数说明（按位置顺序）：

- 第 1 个参数 `modelPath`：模型文件路径，例如 `models\all_9000_320_a1_e30b256s42_04212210.pt`。
- 第 2 个参数 `imageDirectory`：待批量预测的图片目录；会递归读取子目录中的受支持图片。
- 第 3 个参数 `batchSize`：批量推理大小，整数，默认 `128`。越大通常吞吐越高，但显存占用也越高。
- 第 4 个参数 `inferenceImageSize`：推理分辨率，整数，可选；不传时会优先使用模型元数据中的推荐推理尺寸，否则回退到 `320`。
- 第 5 个参数 `enablePreprocessCache`：是否启用推理预处理缓存，`true/false`，默认 `false`。
- 第 6 个参数 `preprocessCacheDirectory`：推理预处理缓存目录，可选；不传时默认使用 `models\inference-cache`。

## 5. CUDA 说明

`Trainer` 和 `Demo` 已引用：

- `TorchSharp`
- `libtorch-cuda-12.8-win-x64`

首次 `restore/build/run` 会下载较大的 CUDA 运行时包。

如果你在自己的应用中只引用 `ImagePopularity.Core`，请在宿主项目里也添加 `libtorch-cuda-12.8-win-x64`。

请确保：

- NVIDIA 驱动正常
- CUDA 兼容（与 `libtorch-cuda-12.8` 匹配）
- `torch.cuda.is_available()` 在运行时返回 `true`
- 当前预处理策略为 `ResizeMode.Pad`（保留完整画面并补边到正方形）

## 6. 5090 参数建议

- `--batch-size`：训练先从 `256` 开始，显存有余量可提升到 `384` 或 `512`
- 推理批大小建议从 `256~1024` 之间压测后选最大稳定值
- 训练分辨率 `--train-image-size` 默认 `320`；推荐推理尺寸自动跟随训练尺寸，但推理时仍可独立覆盖
- `--epochs`：`20~40`
- 程序固定使用 CUDA（不再提供 device 参数）

## 7. 训练耗时与显存预估（1万张图片）

以下为单卡 5090 级别的经验区间，实际取决于磁盘读图速度、CPU 解码能力和 batch 设置：

- `resnet152 + train-image-size 320 + batch-size 256`：
  1 个 epoch 约 `2~6` 分钟，20 epoch 约 `40~120` 分钟
- 显存占用通常在 `14~30 GB` 区间（训练）
- 推理（batch 512~1024）通常在 `4~10 GB` 区间


