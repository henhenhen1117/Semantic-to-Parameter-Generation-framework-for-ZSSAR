# Semantic-to-Parameter-Generation-framework-for-ZSSAR


## Beyond Feature Space: Semantic-to-Parameter Generation for Zero-Shot Skeleton-based Action Recognition：

```markdown
# Semantic-to-Parameter Generation Framework for Zero-Shot Skeleton Action Recognition (PGFA)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

用于零样本骨架动作识别的语义到参数生成框架。本项目实现了一个基于扩散模型的适配器参数生成方法。

## ✨ 主要特性

- **语义驱动的参数生成**：利用文本语义嵌入生成适配器参数
- **扩散模型框架**：使用扩散过程进行参数生成
- **零样本学习**：无需未见类别的训练样本
- **双模态适配器**：同时生成文本和骨架适配器参数
- **高效的推理**：一次生成，多次使用

## 📦 安装

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.3+ (推荐)

### 安装步骤
```bash
# 1. 克隆仓库
git clone https://github.com/henhenhen1117/Semantic-to-Parameter-Generation-framework-for-ZSSAR.git
cd Semantic-to-Parameter-Generation-framework-for-ZSSAR

# 2. 创建conda环境（推荐）
conda create -n pgfa python=3.8
conda activate pgfa

# 3. 安装PyTorch (根据你的CUDA版本)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 4. 安装其他依赖
pip install -r requirements.txt
