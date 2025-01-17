### 数据集

1. 准备数据集

- 通过开源数据集网站或数据集官网进行获取，如[paperswithcode](https://paperswithcode.com/datasets)、[paddlepaddle](https://www.paddlepaddle.org.cn/)等。

- 或自行准备数据。项目提供[crawler.py](./utils/crawler.py)简易爬虫通过搜索引擎获取图片。

  - ```bash
    python crawler.py -q [keyword] -o [Path] -n [numberImages]
    ```

  - **options**:

    - -q --query，搜索关键词
    - -o --output，图片保存目录，`./raw_data/` + `[outfoldname]`
    - -n --number，需要获取的图片数量

  - 爬取后会有部分图片不符合要求，可能需要进一步手动剔除。

2. 预处理数据

- 初始数据文件结构：

  - ```bash
    raw_data/
    ├── [class1Name]
    │   ├── [1].jpg
    │   ├── ...
    │   ├── [n].jpg
    ├──...
    ├── [classnName]
    │   ├── ...
    ```

- 项目提供[dataAugmSplit.py](./utils/dataAugmSplit.py)对数据进行增强扩充和划分为训练集、验证集和测试集。

  - ```bash
    python dataAugmSplit.py -raw [Path] -out [Path] -tr [float] -val [float] -te [float] -tr_times [number] -val_times [number] -te_times [number] -p [float]
    ```

  - **options**:

    - -raw --raw_dir，初始数据所在位置，default='./raw_data'
    - -out --output_dir，输出处理后数据位置，default='./data'
    - -tr --train_ratio，训练集所占初始数据比例，default=0.8
    - -val --val_ratio，验证集所占初始数据比例，default=0.1
    - -te --test_ratio，测试集所占初始数据比例，default=0.1
      - 划分使用`train_test_split()`，随机种子固定为`random_state=42`
    - -tr_times --augment_times_train，训练集增强次数，default=3
    - -val_times --augment_times_val，验证集增强次数，default=1
    - -test_times --augment_times_test，测试集增强次数，default=0

  - -p --prob_aa，使用 AutoAugment 策略增强的概率，default=0.45

- 如果初始数据集过大不想增强扩充，可设置三个增强次数参数为 0
- 划分后文件结构：

  - ```bash
    data/
    ├── test
    │   ├── [class1Name]
    │       ├── [1].jpg
    │       ├── ...
    │       ├── [n].jpg
    │   ├──...
    │   ├── [classnName]
    │       ├── ...
    ├── train
    │       ├── ...
    ├── val
    │       ├── ...
    ```

---

**example：**

```bash
python crawler.py -q 孙悟空 -o wukong -n 10
```

![image-20250111214318556](https://lvyou-article-1325718851.cos.ap-guangzhou.myqcloud.com/imgForTyparo/202501132229895.png)

![image-20250111214528207](https://lvyou-article-1325718851.cos.ap-guangzhou.myqcloud.com/imgForTyparo/202501132229817.png)
