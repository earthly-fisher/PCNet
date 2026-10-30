# PCNet 最小训练与测试集合

网络固定为与原 6 份权重兼容的 ablation_pcnet.PCNet：输入/输出通道 1，depth=5，alpha=0.5，progressive_mode=dual，两个模块均启用。net.py 保留参数名称和计算过程，删除其他网络、消融开关、注释掉的实验代码及无关依赖。

## 安装

建议 Python 3.10 或更新版本，安装与你的 CPU/CUDA 环境匹配的 PyTorch，然后安装其余依赖：

```powershell
pip install -r requirements.txt
```

## 数据接口

准备已处理的灰度图及二值标签，目录为：

```text
data/
  train/image/          train/vessel/
  validation/image/     validation/vessel/
  test/image/           test/vessel/
```

每个 split 的图像和标签文件名相同、尺寸相同；高宽均为 16 的倍数。训练同一 batch 内图像大小相同。数值归一化到 [0,1]，标签按 0.5 二值化。沿用原项目已处理数据的接口。原始数据可用下述 process.py 离线增强；不包含数据集下载或自动划分。

## 训练（预热 → 正式训练 → 每轮验证）

在本文件夹运行：

```powershell
python train.py --data D:/your/data --device cuda:0
```

默认预热 10 轮，正式训练 50 轮。预热使用原 IGLoss，正式训练保留原 MultiLoss 公式，AdamW 和学习率衰减；验证按每图平均计算 accuracy、F1、recall。以验证 F1 保存最佳模型（不再混用 accuracy 与 F1 的保存条件）。

输出 runs/warming.pth、runs/best.pth、runs/last.pth、runs/history.json。完成预热后严格重载最佳预热参数到 module1，再训练完整模型。CPU 可用 --device cpu。

从提供的完整权重继续训练（跳过预热；重新初始化优化器，不是中断续训）：

```powershell
python train.py --data D:/your/data --init model/best.pth
```

使用已有预热权重：

```powershell
python train.py --data D:/your/data --warming-checkpoint D:/warming_model.pth
```

从随机初始化直接训练：

```powershell
python train.py --data D:/your/data --warming-epochs 0
```

## 测试

默认严格加载 model/best.pth：

```powershell
python test.py --data D:/your/data --device cuda:0
```

测试新训练的最佳模型：

```powershell
python test.py --data D:/your/data --checkpoint runs/best.pth
```

输出 results/*.png 概率图和 results/metrics.json。阈值为 0.5。所有权重加载使用 strict=True。

## 原始数据与离线增强

先运行离线增强，然后预热/训练，最后测试：

```powershell
python process.py --source D:/original_data --output D:/processed_data --copies 500 --seed 42
python train.py --data D:/processed_data
python test.py --data D:/processed_data --checkpoint runs/best.pth
```

原始目录为 training、validation、test，每组包含 images 和 vessel（或 label），图像与标签同名。需要事先从训练集划分独立 validation；原代码直接把 test 同时用作 validation，本版不沿用该做法。仅训练集离线增强，验证和测试仅调整尺寸。默认 512×512，标签使用最近邻插值。

本版保留原离线增强的旋转、翻转、20% 高斯加噪及每图 50 份设置，加入固定增强种子。删除原 warp：它没有随机位移场，且 grid_sample 的对齐默认值使其并非可靠的弹性增强。本版将标签与图像统一到处理尺寸，因此测试指标为处理后分辨率的指标。以上修正意味着这不是原实验的逐数值复刻。
