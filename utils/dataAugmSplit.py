"""
@File    : dataAugm.py
@Author  : GiperHsiue
@Time    : 2024/11/7 14:09
"""
import os
import shutil
from PIL import Image
import numpy as np
from PIL import ImageFilter
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import autoaugment
import argparse
from tqdm import tqdm as ProgressBar

# 解析命令行参数
parser = argparse.ArgumentParser(description='Data Augmentation Script')
parser.add_argument('-raw', '--raw_dir', type=str, default='./raw_data', help='Path to the original data directory')
parser.add_argument('-out','--output_dir', type=str, default='./data', help='Path to the output data directory')
parser.add_argument('-tr','--train_ratio', type=float, default=0.8, help='Ratio of training data')
parser.add_argument('-val','--val_ratio', type=float, default=0.1, help='Ratio of validation data')
parser.add_argument('-te','--test_ratio', type=float, default=0.1, help='Ratio of test data')
parser.add_argument('-tr_times','--augment_times_train', type=int, default=3, help='Number of augmentations for training data')
parser.add_argument('-val_times','--augment_times_val', type=int, default=1, help='Number of augmentations for validation data')
parser.add_argument('-test_times','--augment_times_test', type=int, default=0, help='Number of augmentations for test data')
parser.add_argument('-prob_aa','--prob_aa', type=float, default=0.45, help='Probability of using AutoAugment')
args = parser.parse_args()

# 数据增强参数
data_dir = args.raw_dir  # 原始数据目录
output_dir = args.output_dir  # 输出数据目录
train_ratio = args.train_ratio  # 训练集占总数据集的比例
val_ratio = args.val_ratio  # 验证集占总数据集的比例
test_ratio = args.test_ratio  # 测试集占总数据集的比例
augment_times_train = args.augment_times_train  # 训练集增强次数
augment_image_val = args.augment_times_val  # 验证集增强次数
augment_times_test = args.augment_times_test  # 测试集增强次数
prob_aa = args.prob_aa  # AutoAugment的概率

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(os.path.join(output_dir, 'train')):
    os.makedirs(os.path.join(output_dir, 'train'))
if not os.path.exists(os.path.join(output_dir, 'test')):
    os.makedirs(os.path.join(output_dir, 'test'))
if not os.path.exists(os.path.join(output_dir, 'val')):
    os.makedirs(os.path.join(output_dir, 'val'))

# 定义AutoAugment策略
autoaugment_policy = autoaugment.AutoAugmentPolicy.IMAGENET

# 定义增强变换序列
transform = transforms.Compose([
    autoaugment.AutoAugment(autoaugment_policy),
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 数据增强函数-使用AutoAugment
def augment_image_aa(image_path, output_path, idx, transform):
    image = Image.open(image_path).convert('RGB')  # 打开并转换图片为RGB
    augmented_image = transform(image)  # 应用AutoAugment
    augmented_image = augmented_image.permute(1, 2, 0).numpy()  # 转换回PIL图像格式
    pil_image = Image.fromarray((augmented_image * 255).astype('uint8'))  # 转换为PIL图像
    pil_image.save(os.path.join(output_path, f'{idx}.jpg'))  # 保存图像

# 数据增强函数-自定义
def augment_image_mine(image, output_path, idx):
    image = Image.open(image).convert('RGB')  # 打开并转换图片为RGB
    # 随机水平翻转
    if np.random.rand() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # 随机垂直翻转
    if np.random.rand() < 0.1:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # 高斯模糊
    if np.random.rand() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=np.random.rand()))
    # 随机旋转
    angle = np.random.randint(-10, 10)
    image = image.rotate(angle)
    # 随机放大
    if np.random.rand() < 0.5:  # 假设有50%的概率进行放大
        scale_factor = np.random.uniform(1.2, 1.5)  # 随机选择一个放大比例在1.2到1.5之间
        width, height = image.size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        # image = image.resize(new_size, Image.ANTIALIAS)  # 使用ANTIALIAS滤镜进行平滑缩放
        image = image.resize(new_size, Image.LANCZOS)  # 使用ANTIALIAS滤镜进行平滑缩放
        
    # 保存图像
    image.save(os.path.join(output_path, f'{idx}.jpg'))

# 遍历每个类别的文件夹
for class_dir in os.listdir(data_dir):
    print(f"Processing class: {class_dir}")
    class_path = os.path.join(data_dir, class_dir)
    print(f"Processing class path: {class_path}")
    if os.path.isdir(class_path):
        # 获取类别中的所有图片
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.jpg')]
        total_images_cnt = len(images)
        # 将图片先分为tmp集和测试集
        train_images, test_images = train_test_split(images, test_size=int(total_images_cnt * test_ratio), shuffle=True, random_state=42)
        
        # 将tmp集再分为训练集和验证集
        train_images, val_images = train_test_split(train_images, test_size=int(total_images_cnt * val_ratio), shuffle=True, random_state=42)
        # 创建类别子文件夹
        train_class_path = os.path.join(output_dir, 'train', class_dir)
        test_class_path = os.path.join(output_dir, 'test', class_dir)
        val_class_path = os.path.join(output_dir, 'val', class_dir)
        if not os.path.exists(train_class_path):
            os.makedirs(train_class_path)
        if not os.path.exists(test_class_path):
            os.makedirs(test_class_path)
        if not os.path.exists(val_class_path):
            os.makedirs(val_class_path)

        # 复制并增强test集图片
        idx = 1
        print(f"output test class path: {test_class_path}")
        with ProgressBar(total=len(test_images) * (1 + augment_times_test), desc="Processing test images") as pbar:
            for _, img_path in enumerate(test_images, start=1):
                shutil.copy(img_path, os.path.join(test_class_path, f'{idx}.jpg'))
                idx += 1
                pbar.update(1)
                if augment_times_test > 0:
                    for cnt in range(augment_times_test):
                        if np.random.rand() >= (1 - prob_aa):
                            augment_image_aa(img_path, test_class_path, idx, transform)
                        else:
                            augment_image_mine(img_path, test_class_path, idx)
                        idx += 1
                        pbar.update(1)

        # 复制并增强train集图片
        idx = 1
        print(f"output train class path: {train_class_path}")
        with ProgressBar(total=len(train_images) * (1 + augment_times_train), desc="Processing train images") as pbar:
            for _, img_path in enumerate(train_images, start=1):
                shutil.copy(img_path, os.path.join(train_class_path, f'{idx}.jpg'))
                idx += 1
                pbar.update(1)
                for cnt in range(augment_times_train):
                    if np.random.rand() >= (1 - prob_aa):
                        augment_image_aa(img_path, train_class_path, idx, transform)
                    else:
                        augment_image_mine(img_path, train_class_path, idx)
                    idx += 1
                    pbar.update(1)

        # 复制并增强val集图片
        idx = 1
        print(f"output val class path: {val_class_path}")
        with ProgressBar(total=len(val_images) * (1 + augment_image_val), desc="Processing val images") as pbar:
            for _, img_path in enumerate(val_images, start=1):
                shutil.copy(img_path, os.path.join(val_class_path, f'{idx}.jpg'))
                idx += 1
                pbar.update(1)
                if augment_image_val > 0:
                    for cnt in range(augment_image_val):
                        if np.random.rand() >= (1 - prob_aa):
                            augment_image_aa(img_path, val_class_path, idx, transform)
                        else:
                            augment_image_mine(img_path, val_class_path, idx)
                        idx += 1
                        pbar.update(1)

print("数据增强和分配完成。")