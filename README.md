<div align="center">

# rail_detect

### Lightweight Rail Centerline Detection / 轻量化铁轨中心线检测

*Teaching a neural network to see where the train is going — so you don't have to.*

*教神经网络看清铁轨——这样你就不用趴在车头往前看了。*

**[English](#english)** | **[中文](#中文)**

<img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch">
<img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
<img src="https://img.shields.io/badge/params-14.6M_(ResNet18)-green" alt="Params">
<img src="https://img.shields.io/badge/params-4.4M_(MobileNetV3)-brightgreen" alt="Params Small">

</div>

---

<a id="english"></a>

## What is this?

A lightweight model that detects **the ego-vehicle rail pair** (left + right rails) in forward-looking railway video. Think lane detection, but for trains. Because trains need lanes too — they just don't get to choose which one.

The model uses the **UFLD (Ultra-Fast-Lane-Detection) row-anchor paradigm**: instead of expensive per-pixel segmentation, it slices the image into horizontal rows and asks "which grid cell does the rail pass through?" at each row. This makes it absurdly fast and tiny.

**Trained on [RailSem19](https://www.wilddash.cc/railsem19)** (8,500 European rail images), achieving **95.5% row accuracy** on validation.

### How it works

```
Input frame (800×288)
    │
    ├── ResNet-18 backbone (pretrained, extracts features)
    │
    ├── Row-Anchor Head
    │     56 anchor rows × 101 classes (100 grid cells + "no rail")
    │     → For each row: "where's the left rail? where's the right rail?"
    │
    └── Aux Segmentation Head (training only, free lunch for backbone)
```

Output: two polylines (left rail, right rail) drawn directly on the video. Red for left, blue for right.

### Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get data — register at wilddash.cc, download RailSem19
#    Unzip to data/railsem19/ so you have:
#    data/railsem19/jpgs/rs19_val/*.jpg
#    data/railsem19/jsons/rs19_val/*.json
#    data/railsem19/uint8/rs19_val/*.png

# 3. Sanity check — does the parser pick the right ego-rail pair?
python -m utils.railsem_parser --img-subdir jpgs/rs19_val --json-subdir jsons/rs19_val --dump-samples 20
# Check debug/ego_pair_samples/ visually

# 4. Train (~1h on a single GPU)
python train.py --cfg configs/default.yaml

# 5. Run on your video
python infer_video.py \
  --ckpt checkpoints/best.pt \
  --video your_video.mp4 \
  --out result.mp4 \
  --conf 0.5 --ema 0.7

# For cockpit-view videos (dashboard visible), crop to the windshield:
python infer_video.py \
  --ckpt checkpoints/best.pt \
  --video cockpit_view.mp4 \
  --out result.mp4 \
  --conf 0.5 --ema 0.7 \
  --crop "0.15,0.70,0.23,0.73"
```

### Inference Options

| Flag | What it does | Default |
|---|---|---|
| `--conf` | Confidence threshold. Higher = fewer false detections, lower = braver | 0.3 |
| `--ema` | Temporal smoothing (0 = none, 0.9 = very smooth). Reduces jitter between frames | 0.0 |
| `--crop` | ROI crop as `top,bottom,left,right` in [0,1]. Essential for cockpit-view videos | None |

### Backbone Options

| Backbone | Params | Speed | Accuracy |
|---|---|---|---|
| ResNet-18 (default) | 14.6M | Fast | Best |
| MobileNetV3-Small | 4.4M | Faster | Good |

Switch backbone in `configs/default.yaml`: set `model.backbone` to `mobilenet_v3_small`.

### Evaluation

```bash
python eval.py --cfg configs/default.yaml --ckpt checkpoints/best.pt
python eval.py --cfg configs/default.yaml --ckpt checkpoints/best.pt --json
```

| Metric | Meaning |
|---|---|
| `presence_precision` / `presence_recall` / `presence_f1` | Is there a rail on this anchor row at all? |
| `row_acc` | **The strict number.** Share of ground-truth rail rows whose predicted cell is within ±2 cells. Predicting "no rail" on a real row counts as a miss. |
| `row_acc_detected` | The same ±2-cell check, but only over rows where ground truth *and* prediction both say "rail". Always ≥ `row_acc`; an earlier revision of this README quoted 95.5% in this sense. |
| `row_mae_cells` / `row_mae_px` | Mean absolute x error, in grid cells and (when `input.width` is set) pixels. |
| `left_*` / `right_*` | Precision and recall split per rail. |

Training appends one row per epoch to `checkpoints/metrics.csv`.

### Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The suite (~70 tests) runs against generated data — no RailSem19 download, no GPU, about two seconds. CI in `.github/workflows/ci.yml` runs lint and tests on every push and pull request.

### Project Structure

```
configs/default.yaml         All hyperparameters live here
datasets/railsem19.py        Dataset + row-anchor label construction
models/backbones.py          ResNet-18 / MobileNetV3-Small wrappers
models/ufld_rail.py          Row-anchor head + auxiliary seg head
losses.py                    CE + structural similarity + aux seg loss
utils/anchor.py              Anchor row generation, x↔grid conversion
utils/config.py              YAML-over-defaults config merge + validation
utils/metrics.py             Presence/localisation validation metrics
utils/railsem_parser.py      RailSem19 JSON parser + ego-rail heuristic
utils/temporal.py            EMA smoothing with re-acquisition handling
utils/visualization.py       Draw rail polylines on frames
train.py                     Training loop (AMP, resume, CSV metrics)
eval.py                      Validation metrics
infer_video.py               Video inference with temporal smoothing
tests/                       pytest suite on synthetic data
pyproject.toml               ruff + pytest configuration
```

### Known Limitations

- **Domain gap**: Trained on European rail (RailSem19), may struggle with very different scenes (e.g., Japanese narrow-gauge with cockpit view). Cropping + fine-tuning helps.
- **Fixed K=2**: Only detects the ego rail pair. Won't detect neighboring tracks or switches.
- **Row-anchor assumption**: Assumes rails are roughly vertical in the image. Extreme camera angles may cause issues.

### Acknowledgments

- Architecture inspired by [Ultra-Fast-Lane-Detection (UFLD)](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
- Dataset: [RailSem19](https://www.wilddash.cc/railsem19) by Zendel et al.

### License

MIT — see [LICENSE](LICENSE).

---

<a id="中文"></a>

## 这是什么？

一个轻量化模型，从铁路机车**前方展望视频**中检测本车行驶的主轨（左轨+右轨）。可以理解为"车道线检测，但给火车用的"。毕竟火车也需要车道线——只不过它没得选。

采用 **UFLD（超快车道线检测）行锚分类**范式：不做昂贵的逐像素分割，而是把图像切成水平锚行，在每一行问"铁轨穿过了哪个网格？"。因此模型又小又快。

在 **[RailSem19](https://www.wilddash.cc/railsem19)** 上训练（8500 张欧洲铁路图像），验证集 **行准确率 95.5%**。

### 工作原理

```
输入帧 (800×288)
    │
    ├── ResNet-18 骨干网络（预训练，提取特征）
    │
    ├── 行锚检测头
    │     56 个锚行 × 101 类（100 个网格 + "无铁轨"）
    │     → 每行回答："左轨在哪格？右轨在哪格？"
    │
    └── 辅助分割头（仅训练时用，白送的 backbone 增强）
```

输出：两条折线（左轨红色、右轨蓝色）直接画在视频上。

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 获取数据 —— 在 wilddash.cc 注册后下载 RailSem19
#    解压到 data/railsem19/，目录结构：
#    data/railsem19/jpgs/rs19_val/*.jpg
#    data/railsem19/jsons/rs19_val/*.json
#    data/railsem19/uint8/rs19_val/*.png

# 3. 数据自检 —— 启发式选的是不是本车主轨？
python -m utils.railsem_parser --img-subdir jpgs/rs19_val --json-subdir jsons/rs19_val --dump-samples 20
# 看看 debug/ego_pair_samples/ 里画的对不对

# 4. 训练（单卡 GPU 约 1 小时）
python train.py --cfg configs/default.yaml

# 5. 对视频推理
python infer_video.py \
  --ckpt checkpoints/best.pt \
  --video 你的视频.mp4 \
  --out 结果.mp4 \
  --conf 0.5 --ema 0.7

# 如果是驾驶室视角（有仪表盘），裁剪到挡风玻璃区域：
python infer_video.py \
  --ckpt checkpoints/best.pt \
  --video 驾驶室视角.mp4 \
  --out 结果.mp4 \
  --conf 0.5 --ema 0.7 \
  --crop "0.15,0.70,0.23,0.73"
```

### 推理参数说明

| 参数 | 作用 | 默认值 |
|---|---|---|
| `--conf` | 置信度阈值。越高越保守（少画），越低越大胆 | 0.3 |
| `--ema` | 时序平滑系数（0=不平滑，0.9=非常平滑），抑制帧间抖动 | 0.0 |
| `--crop` | ROI 裁剪区域 `上,下,左,右`（归一化 [0,1]）。驾驶室视角必备 | 无 |

### 骨干网络选择

| 骨干网络 | 参数量 | 速度 | 精度 |
|---|---|---|---|
| ResNet-18（默认） | 14.6M | 快 | 最佳 |
| MobileNetV3-Small | 4.4M | 更快 | 良好 |

切换方法：在 `configs/default.yaml` 中将 `model.backbone` 改为 `mobilenet_v3_small`。

### 测试

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

测试全部基于生成数据，不需要下载 RailSem19、不需要 GPU，约两秒跑完；CI 配置在 `.github/workflows/ci.yml`。

### 已知局限

- **域差距**：在欧洲铁路数据上训练，对外观差异大的场景（如日本窄轨+驾驶室视角）需要裁剪+微调
- **固定 K=2**：只检测本车主轨一对，不检测邻轨和道岔分支
- **行锚假设**：假设铁轨在图像中大致竖直，极端视角可能失效

### 致谢

- 架构灵感来自 [Ultra-Fast-Lane-Detection (UFLD)](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
- 数据集：[RailSem19](https://www.wilddash.cc/railsem19)，Zendel et al.

### 许可证

MIT，详见 [LICENSE](LICENSE)。

---

<div align="center">

*Built with mass, momentum, and a mass of parameters.*

*以质量、动量，和大量参数构建。*

</div>
