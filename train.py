"""
@File    : train.py
@Author  : GiperHsiue
@Time    : 2024/11/7 14:43
"""
import random
import numpy
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import logging
from modelForClassify import Classifier
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import torch.distributed as dist
import pynvml

Image.MAX_IMAGE_PIXELS = None

def setup_logging():
    # Set up logging to file
    logger = logging.getLogger('TrainCarLogger')
    logger.setLevel(logging.INFO)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/train_{current_time}.log'
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def center_crop(image):
    # Center crop the image to the specified output size
    image_width, image_height = image.size
    crop_size = min(image_width, image_height)
    left = int((image_width - crop_size) // 2)
    top = int((image_height - crop_size) // 2)
    return image.crop((left, top, left + crop_size, top + crop_size))

def get_transforms():
    # Define image transformations
    CenterCrop = transforms.Lambda(lambda image: center_crop(image))
    transform = transforms.Compose([
        CenterCrop,
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform

def get_data_loaders(train_dir, test_dir, batch_size):
    # Create data loaders for training and validation datasets
    transform = get_transforms()
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True, pin_memory=True, num_workers=4)
    val_dataset = ImageFolder(root=test_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size[1], shuffle=False)
    return train_loader, val_loader



# def initialize_model(num_gpus, num_classes, resume):
#     # Initialize the model and move it to the appropriate device
#     if torch.cuda.device_count() < num_gpus: # need > had
#         print("nomore GPUs!")
#         exit()
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Classifier(num_classes)
#     if resume == '':
#         model.to(device)
#     else:
#         print("resume training!")
#         logger.info("resume training!")
#         model = Classifier(num_classes)
#         checkpoint = torch.load(resume)
#         for (k, v) in checkpoint.items():
#             if k.startswith('module.'):
#                 # 移除键名前的'module.'前缀,模型在训练时候使用了多GPU
#                 new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
#                 model.load_state_dict(new_state_dict)
#                 break
#             else:
#                 model.load_state_dict(checkpoint)
#                 break
#         model.to(device)
#     # jugle device
#     if num_gpus == 0 or device.type == 'cpu':
#         print("just use cpu!")
#         device = torch.device("cpu")
#         model.to(device)
#     else: # need gpus >= 1
#         # torch.cuda.device_count() >= num_gpus:
#         free_gpus = []
#         for i in range(torch.cuda.device_count()):
#             handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#             info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#             gpu_memory = info.total
#             gpu_reserved = info.used
#             print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#             # gpu_memory = torch.cuda.get_device_properties(i).total_memory
#             # gpu_reserved = torch.cuda.memory_reserved(i)
#             gpu_free = gpu_memory - gpu_reserved
#             print(f"GPU {i} memory: {gpu_free / 1024 ** 3:.2f} GB free out of {gpu_memory / 1024 ** 3:.2f} GB") 
#             if gpu_free > gpu_memory / 2: # 大于一半内存
#                 free_gpus.append(i)
#         if len(free_gpus) < num_gpus:
#             print("Not enough free GPUs available!")
#             exit()
#         print(torch.cuda.device_count())
#         print(f"can using GPUs: {free_gpus[:num_gpus]}")
#         model = nn.DataParallel(model, device_ids=free_gpus[:num_gpus])
#     return model, device

def initialize_model(num_gpus, num_classes, resume):
    if num_gpus > 0: 
        pynvml.nvmlInit()
    
    # 检查 GPU 数量是否足够
    if torch.cuda.device_count() < num_gpus:
        print("Not enough GPUs available!")
        exit()

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = Classifier(num_classes)

    # 加载预训练模型（如果有）
    if resume:
        print("Resuming training from checkpoint...")
        checkpoint = torch.load(resume, map_location=device)  # 加载到当前设备
        if any(k.startswith('module.') for k in checkpoint.keys()):
            # 如果 checkpoint 的键名以 'module.' 开头，移除前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
    

    # 多 GPU 设置
    if num_gpus >= 1:
        # 检查可用 GPU
        free_gpus = []
        try:
            pynvml.nvmlInit()
            for i in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = info.total
                gpu_used = info.used
                gpu_free = gpu_memory - gpu_used
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"GPU {i} memory: {gpu_free / 1024 ** 3:.2f} GB free out of {gpu_memory / 1024 ** 3:.2f} GB")
                if gpu_free > gpu_memory / 2:  # 大于一半内存
                    free_gpus.append(i)
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            print(f"Failed to initialize pynvml: {e}")
            free_gpus = list(range(torch.cuda.device_count()))  # 如果 pynvml 失败，使用所有 GPU

        if len(free_gpus) < num_gpus:
            print("Not enough free GPUs available!")
            exit()

        print(f"Using GPUs: {free_gpus[:num_gpus]}")
        model = nn.DataParallel(model, device_ids=free_gpus[:num_gpus])
        device = torch.device(f"cuda:{free_gpus[0]}")  # 主设备设置为第一个 GPU
    model.to(device)
    return model, device

def train_model(model, device, train_loader, val_loader, val_dataset, num_epochs, logger, resume):
    # Train the model and log the training process
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    loss_list = []
    test_loss_list = []
    accuracy_list = []
    model_path = 'save_model/model.pth'
    best_acc = 0
    best_epoch = 0
    # check resume
    if resume != '':
        print("for resume checkpoint val:")
        logger.info("for resume checkpoint val:")
        model.eval()
        test_loss_tmp = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss_tmp += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / len(val_dataset)
        best_acc = accuracy
        print(f'val Loss: {test_loss_tmp:.4f}, val Accuracy: {accuracy:.2f}%')
        logger.info(f'val Loss: {test_loss_tmp:.4f}, val Accuracy: {accuracy:.2f}%')
        print(f'continue training')
        logger.info(f'continue training')
    # Train the model
    for epoch in range(num_epochs):
        epoch = epoch + 1
        model.train()
        train_loss_tmp = 0
        pbar = tqdm(total=len(train_loader))
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss_tmp += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            pbar.update(1)
        pbar.close()

        loss_list.append(train_loss_tmp / len(train_loader))
        my_lr_scheduler.step()

        model.eval()
        test_loss_tmp = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss_tmp += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        test_loss_list.append(test_loss_tmp / len(val_loader))
        accuracy = 100 * correct / len(val_dataset)
        accuracy_list.append(accuracy)

        if accuracy > best_acc:
            print(f'last best_acc: {best_acc:.2f}, current best_acc: {accuracy:.2f}')
            logger.info(f'last best_acc: {best_acc:.2f}, current best_acc: {accuracy:.2f}')
            best_acc = accuracy
            best_epoch = epoch
            # if os.path.exists(model_path):
            #     current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            #     model_path = f'save_model/model_{current_time}.pth'
            torch.save(model.state_dict(), model_path)
            logger.info(f'Model saved to {model_path}')
            print(f'Model saved to {model_path}')
        logger.info(f'Epoch: {epoch}, Train Loss: {loss_list[-1]:.4f}, val Loss: {test_loss_list[-1]:.4f}, val Accuracy: {accuracy:.2f}%')
        print(f'Epoch: {epoch}, Train Loss: {loss_list[-1]:.4f}, val Loss: {test_loss_list[-1]:.4f}, val Accuracy: {accuracy:.2f}%')

    return loss_list, test_loss_list, accuracy_list

def plot_results(loss_list, test_loss_list, accuracy_list):
    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    epochs = range(1, len(loss_list) + 1)  # Modify this line
    plt.plot(epochs, loss_list, label='Training Loss')  # Modify this line
    plt.plot(epochs, test_loss_list, label='val Loss')  # Modify this line
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and val Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_list, label='val Accuracy')  # Modify this line
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('val Accuracy')
    plt.legend()

    plt.tight_layout()
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'pics/loss_acc_{current_time}.png')
    plt.show()

def set_seed(seed):
    print(f"Setting random seed to {seed}")
    # 设置随机种子
    torch.backends.cudnn.enabled = True  # pytorch 使用CUDANN 加速，即使用GPU加速
    torch.backends.cudnn.benchmark = False  # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False 则每次的算法一致
    torch.backends.cudnn.deterministic = True  # 设置每次返回的卷积算法是一致的
    torch.manual_seed(seed)  # 为当前CPU 设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前的GPU 设置随机种子
    torch.cuda.manual_seed_all(seed)  # 当使用多块GPU 时，均设置随机种子
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

 
def is_dist_avail_and_initialized():
    # 判断当前环境中是否支持分布式训练
    if not dist.is_available():
        return False
    # 检查当前环境是否已经成功初始化了分布式训练环境
    if not dist.is_initialized():
        return False
    return True
 
def get_rank():
    # 判断分布式训练是否可用且是否已成功初始化
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

if __name__ == "__main__":
    logger = setup_logging()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a classification model.')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('-tr', '--train_dir', type=str, default='./data/train', help='Path to the training dataset')
    parser.add_argument('-val', '--val_dir', type=str, default='./data/val', help='Path to the val dataset')
    parser.add_argument('-g', '--gpus', type=int, default=0, help='Number of GPUs to use')
    parser.add_argument('-batch', '--batch_size', type=int, nargs=2, default=[32, 1], help='Batch sizes for training and validation') # example: -batch 32 1
    parser.add_argument('-resume', '--resume', type=str, default='', help='Resume training from a checkpoint') # ./save_model/model.pth
    parser.add_argument('-sd', '--seed', type=int, default=1024, help='Random seed')
    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        logger.info(f'{arg}: {getattr(args, arg)}')
    
    if (args.seed != -1):
        set_seed(args.seed + get_rank())

    train_loader, val_loader = get_data_loaders(args.train_dir, args.val_dir, args.batch_size)
    model, device = initialize_model(args.gpus, len(train_loader.dataset.classes), args.resume)
    loss_list, test_loss_list, accuracy_list = train_model(model, device, train_loader, val_loader, val_loader.dataset, args.epochs, logger, args.resume)
    plot_results(loss_list, test_loss_list, accuracy_list)
