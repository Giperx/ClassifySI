<div align="center">
  <img src="https://lvyou-article-1325718851.cos.ap-guangzhou.myqcloud.com/imgForTyparo/202501132238376.png" alt="Description" width="260" />
</div>

# 图像分类通用简易框架 - ClassifySI

实现一个简易框架，用于进行深度学习图像分类的入门探索。

可通过提前准备或爬取的数据集和自定义的网络结构进行训练及测试。

## 准备

- 数据集
- 环境
- 训练网络搭建

### 数据集

详见：[GetDataStart.md](./utils/GetDataStart.md)

### 环境

1. 安装 python。

```bash
conda create -n classifysi python=3.8 -y
conda activate classifysi
```

2. 安装 pytorch、torchvision，下为示例。

```bash
# 换源（可选）：
#conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
#conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
#conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# for linux
#conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# for legacy win-64
#conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
# 安装pytorch：
# OnlyCPU
#  conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
#  pip3 install torch torchvision torchaudio
# GPU: 在https://pytorch.org/get-started/previous-versions/中查看符合cuda版本进行安装
#example1: conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3
# pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#example2: pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

进入项目目录，安装其它依赖。

```bash
pip3 install -r requirements.txt
```

### 训练网络搭建

在[modelForClassify.py](./modelForClassify.py)中定义模型结构，或使用注释中提供的两个预训练网络模型：ResNet-18、MobileNetV2

```python
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # 定义模型

    def forward(self, x):
        # 定义前向传播
```

**注：** 在训练阶段，`resize`为 $` 448 \times 448 `$ 。

- 如需自定义最后的图片大小，需在[train.py](https://github.com/Giperx/ClassifySI/blob/11392bd53e3ad854d81098183a4249d1247c1321/train.py#L56)和[test.py](https://github.com/Giperx/ClassifySI/blob/11392bd53e3ad854d81098183a4249d1247c1321/test.py#L52)中修改`get_transforms()`函数的`transforms.Resize((448, 448))`。

## 训练及测试

### 训练

下面三行命令等价（默认参数）

- ```bash
  python train.py
  ```

- ```bash
  python train.py -e 50 -tr ./data/train -val ./data/val -g 0 -batch 32 1 -sd -1
  ```

- ```bash
  python train.py --epochs 50 --train_dir ./data/train --val_dir ./data/val --gpus 0 --batch_size 32 1 --seed -1
  ```

- **options**:

  - -e --epochs，训练轮数，default=50

  - -tr --train_dir，训练集路径，default='./data/train'

  - -val --val_dir，验证集路径，default='./data/val'

  - -g --gpus，使用 GPU 数量，default=0，0 表示使用 CPU 进行训练

  - -batch --batch_size，训练和验证时的批次大小，default=[32, 1]

  - -sd --seed，随机数种子，default=-1，-1 表示不进行随机数种子的固定

  - -resume --resume，模型权重文件路径，default=''。可从路径中加载权重继续训练

    - example：

    - ```bassh
      python train.py -resume ./save_model/model.pth
      ```

训练结束时，`log/`中保存输出日志信息；`/pics`中保存训练的 loss、acc 曲线；`./save_model/model.pth`中保存训练过程中达到 best_acc 的模型权重参数。

**如果报错（Linux）：**

```bash
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```

可在命令行运行`unset LD_LIBRARY_PATH`解决。参考[issues](https://github.com/coqui-ai/TTS/issues/1517#issuecomment-1402024354)

### 测试

下面三行命令等价（默认参数）

- ```bash
  python test.py
  ```

- ```bash
  python test.py -t ./data/test -batch 1 -m ./save_model/model.pth
  ```

- ```bash
  python test.py -test_dir ./data/test --batch_size 1 --model_path ./save_model/model.pth
  ```

- **options**:

  - -t --test_dir，测试集路径，default='./data/test'
  - -batch --batch_size，测试时的批次大小，default=1
  - -m --model_path，模型权重文件路径，default='./save_model/model.pth'

测试结束时，`log/`中保存输出日志信息（模型准确率 Accuracy、F1 Score，各类精确率 Precision、召回率 Recall、特异度 Specificity、F1 Score，模型参数量，推理速度）；`/pics`中保存测试生成的混淆矩阵图；`./save_model/model_all.model`保存整个模型（可导入[Netron](https://github.com/lutzroeder/netron)进行可视化）。

---

[新能源汽车图像分类应用实践](./example_classify_car/新能源汽车图像分类实践.md)
