# Siamese Registration Mamba 训练指南

## 项目概述
使用 **SiameseRegistMamba** 模型和 **DCOS_SSIM** 损失函数进行SAR/OPT图像配准任务训练。

## 文件结构
```
e:\L2regist\
├── train.py                    # 简化版训练脚本（直接运行）
├── train_regist_dcos.py        # 完整版训练脚本（支持命令行参数）
├── model_build/
│   ├── siamese_regist_mamba.py # 主模型架构
│   └── datasets.py            # 数据集加载器
├── loss/
│   └── dcos_ssim.py           # DCOS_SSIM损失函数
└── dataset/
    └── Train/512/
        ├── SAR/               # SAR图像（00000.png-00096.png）
        └── OPT/               # OPT图像（00000.png-00096.png）
    ..../256



```

## 快速开始

### 方法1：简化训练（推荐）
```bash
python train.py
```

### 方法2：完整训练（可自定义参数）
```bash
python train_regist_dcos.py --epochs 50 --batch_size 4 --lr 1e-4
```

## 训练参数说明

### train_regist_dcos.py 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | 'dataset' | 数据集根目录 |
| `--image_size` | 512 | 输入图像尺寸 |
| `--batch_size` | 4 | 批次大小（根据GPU内存调整） |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--margin` | 1.0 | 损失中的margin参数 |
| `--save_dir` | 'checkpoints' | 模型保存目录 |

## 模型架构

### SiameseRegistMamba
- **输入**: 单通道SAR/OPT图像 [1, 512, 512]
- **输出**: 256维特征嵌入向量
- **结构**:
  - 基于SiameseMambaNet的孪生架构
  - 集成GlobalRegistrationBlock和LocalRegistrationBlock
  - 自动权重初始化

### DCOS_SSIM损失函数
- **功能**: 最小化相似图像对的嵌入距离
- **特点**: 结合对比学习和SSIM损失
- **优化目标**: 使配准图像对在嵌入空间中距离更近

## 训练输出

### 生成的文件
- `best_regist_model.pth` - 最佳验证损失模型
- `final_regist_model.pth` - 最终训练模型
- `checkpoints/` - 训练检查点目录

### 控制台输出示例
```
🚀 Siamese Registration Mamba 训练开始
使用设备: cuda
📂 正在加载数据集...
✅ 数据集加载完成，共 97 个样本
🏗️ 正在创建SiameseRegistMamba模型...
📊 模型参数量: 1,234,567
🎯 开始训练...
✨ Epoch  1/20 完成 | 平均损失: 0.1234 | 耗时: 45.2s
🏆 保存最佳模型 (损失: 0.1234)
```

## 使用训练好的模型

```python
import torch
from model_build.siamese_regist_mamba import SiameseRegistMamba

# 加载模型
model = SiameseRegistMamba()
model.load_state_dict(torch.load('best_regist_model.pth'))
model.eval()

# 使用模型
sar_image = ...  # [1, 512, 512]
opt_image = ...  # [1, 512, 512]
sar_embedding = model.embed(sar_image.unsqueeze(0))
opt_embedding = model.embed(opt_image.unsqueeze(0))
```

## 故障排除

### 常见问题
1. **CUDA内存不足**
   - 解决：减小batch_size（如 `--batch_size 2`）
   
2. **数据集加载失败**
   - 检查文件命名格式是否为 `00000.png` 到 `00096.png`
   - 确保SAR和OPT文件夹中文件数量匹配

3. **训练速度慢**
   - 使用GPU: `python train.py --device cuda`
   - 减少数据增强: 修改 `augment=True` 为 `augment=False`

### 性能优化建议
```bash
# 使用GPU训练
python train_regist_dcos.py --device cuda

# 小批量训练（适合小显存）
python train_regist_dcos.py --batch_size 2 --epochs 100

# 快速测试
python train.py  # 20个epoch快速验证
```

## 数据集要求

### 文件命名规范
- SAR图像: `dataset/Train/512/SAR/00000.png` - `00096.png`
- OPT图像: `dataset/Train/512/OPT/00000.png` - `00096.png`

### 图像格式
- 格式: PNG
- 通道: 单通道灰度图
- 尺寸: 512×512像素

## 训练监控

### 损失曲线解读
- **训练损失下降**: 模型正在学习
- **验证损失上升**: 可能过拟合，减少epoch
- **损失震荡**: 尝试减小学习率

### 训练时间估算
- CPU: ~5-10分钟/epoch
- GPU: ~30-60秒/epoch
- 97张图像，batch_size=4
