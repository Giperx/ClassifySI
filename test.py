"""
@File    : test.py
@Author  : GiperHsiue
@Time    : 2024/11/7 18:38
"""
import argparse
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from modelForClassify import Classifier
import logging
import os
from prettytable import PrettyTable
import time  # 导入time模块

def setup_logger():
    # 配置日志
    logger = logging.getLogger('TestCarLogger')
    logger.setLevel(logging.INFO)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/test_{current_time}.log'
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def center_crop(image):
    # 自定义中心裁剪函数
    image_width, image_height = image.size
    crop_size = min(image_width, image_height)  # 计算最大的正方形尺寸
    left = int((image_width - crop_size) // 2)  # 计算左上角的横坐标
    top = int((image_height - crop_size) // 2)  # 计算左上角的纵坐标
    return image.crop((left, top, left + crop_size, top + crop_size))  # 进行中心裁剪

def get_transform():
    # 将自定义的中心裁剪函数和放缩函数包装成transforms
    CenterCrop = transforms.Lambda(lambda image: center_crop(image))
    return transforms.Compose([
        CenterCrop,  # 中心裁剪为最大的正方形
        transforms.Resize((448, 448)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化
    ])

def load_model(model_path, device, num_classes):
    model = Classifier(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    # torch >= 1.13
    # state_dict = torch.load(model_path, map_location=device, weights_only=True)
    for (k, v) in state_dict.items():
        if k.startswith('module.'):
            # 移除键名前的'module.'前缀,模型在训练时候使用了多GPU
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            break
        else:
            model.load_state_dict(state_dict)
            break
    torch.save(model, './save_model/model_all.model')
    model.eval()  # 设置为评估模式
    model.to(device)
    return model

def test_model(model, test_loader, device, classes, logger):
    all_labels = []
    all_preds = []
    all_probs = []
    inference_times = []  # 用于存储每次推理的时间

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for images, labels in test_loader:
            start_time = time.time()  # 记录开始时间
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            end_time = time.time()  # 记录结束时间
            pbar.update(1)
            inference_times.append(end_time - start_time)  # 记录每次推理的时间

            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())
            all_probs.extend(probabilities.cpu().numpy())
        pbar.close()
    return all_labels, all_preds, inference_times

def calculate_metrics(all_labels, all_preds, inference_times, classes, logger):
    # 计算平均推理时间，只计算后一半数据的平均时间
    inference_speed = np.mean(inference_times[len(inference_times) // 2:])

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'pics/ConfusionMatrix_test_{current_time}.png')
    plt.show()




    # calculate accuracy
    sum_TP = 0
    for i in range(len(classes)):
        sum_TP += cm[i, i]				# 对角线元素求和
    acc = sum_TP / np.sum(cm)
    # print("the model accuracy is ", acc)
    # logger.info(f"the model accuracy is {acc}")

    # precision, recall, specificity, F1 Score
    table = PrettyTable()
    table.field_names = ["", "Precision", "Recall", "Specificity", "F1 Score"]	# 第一个元素是类别标签
    for i in range(len(classes)):			# 针对每个类别进行计算
        # 整合其他行列为不属于该类的情况
        TP = cm[i, i]
        FP = np.sum(cm[i, :]) - TP
        FN = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - TP - FP - FN
        Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.		# 注意分母为 0 的情况
        Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
        Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
        # 计算F1 Score
        F1 = round(2 * (Precision * Recall) / (Precision + Recall), 3) if (Precision + Recall) != 0 else 0.
        table.add_row([classes[i], Precision, Recall, Specificity, F1])
    print(table)
    logger.info(table)


    # 计算每个类别的准确率
    # class_accuracies = []
    # for i in range(len(classes)):
    #     TP = cm[i, i]
    #     FN = np.sum(cm[:, i]) - TP
    #     class_acc = TP / (TP + FN + 0.0001)  # +0.0001 to avoid division by zero
    #     class_accuracies.append(class_acc)
    #     print(f'Accuracy for {classes[i]}: {class_acc:.3f}')
    #     logger.info(f'Accuracy for {classes[i]}: {class_acc:.3f}')
    # avg_accuracy = np.mean(class_accuracies)
    return inference_speed, acc

def log_metrics(logger, model, inference_speed, avg_accuracy):
    # 记录平均准确率
    print(f'model Accuracy: {avg_accuracy:.3f}')
    logger.info(f'model Accuracy: {avg_accuracy:.3f}')
    # 计算模型参数量
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters in the model: {pytorch_total_params}')
    logger.info(f'Total parameters in the model: {pytorch_total_params}')

    # 计算推理速度（每秒处理的图片数）
    print(f'Inference speed: {1/inference_speed:.3f} images/second')
    logger.info(f'Inference speed: {1/inference_speed:.3f} images/second')

    # 计算延迟
    print(f'Latency: {inference_speed:.3f} seconds')
    logger.info(f'Latency: {inference_speed:.3f} seconds')

    
def main():
    logger = setup_logger()
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Test a classification model.')
    parser.add_argument('-t', '--test_dir', type=str, default='./data/test', help='Path to the test dataset')
    parser.add_argument('-batch', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-m', '--model_path', type=str, default='./save_model/model.pth', help='Path to the model file')
    args = parser.parse_args()
    test_dir = args.test_dir
    batch_size = args.batch_size
    model_path = args.model_path

    transform_test = get_transform()

    test_dataset = ImageFolder(root=test_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, len(test_dataset.classes))

    classes = test_dataset.classes

    all_labels, all_preds, inference_times = test_model(model, test_loader, device, classes, logger)

    inference_speed, avg_accuracy = calculate_metrics(all_labels, all_preds, inference_times, classes, logger)

    log_metrics(logger, model, inference_speed, avg_accuracy)


if __name__ == '__main__':
    main()
