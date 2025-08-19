import os
import sys
from datetime import datetime
from ultralytics import YOLO
from create_datasets import convert_dataset

def train(yaml_path, imgsz, batch, device):
    # 1. 初始化模型
    model = YOLO(model_path)  # 加载预训练权重
    
    # 2. 训练参数配置
    train_args = {
        'data': yaml_path,       # 数据集配置文件路径
        'imgsz': imgsz,           # 图像尺寸
        'batch': batch,            # 批次大小
        'epochs': 150,           # 训练轮次
        'device': device,  # 自动选择设备
        'lr0': 1e-4,           # 初始学习率
        'weight_decay': 0.0005, # 权重衰减
        'augment': True,        # 启用内置数据增强
        'close_mosaic': 20,         # 马赛克增强概率
        'mixup': 0.01           # MixUp增强概率
    }
    
    # 3. 开始训练
    results = model.train(**train_args)
    return results

if __name__ == "__main__":

    # yaml_path = sys.argv[1]
    # device = sys.argv[1]
    # model_path = sys.argv[2]
    # imgsz = int(sys.argv[4])
    # batch = int(sys.argv[5])

    '''
    else:
        # 获取命令行参数
        input_dir = sys.argv[1]    # 输入目录（包含XML和图像）
        output_dir = sys.argv[2]   # 输出目录
        class_name = sys.argv[3]   # 类别名称
    
        # 生成数据集（假设有create_dataset.py）
        from create_dataset import dataset_main
        yaml_path = dataset_main(input_dir, output_dir, class_name)
    '''
    # 记录训练时间

    input_dir = r"C:\Users\developer\Desktop\fzc_ts\03"
    output_dir = r'C:\Users\developer\Desktop\fzc_ts\output'
    class_names = ['fzc']

    yaml_path = convert_dataset(input_dir, output_dir, class_names,
    fs_trans = True,
    split_five_equal = True)

    start_time = datetime.now()
    print(f"训练开始时间: {start_time}")
    device = 0
    model_path = r"C:\Users\developer\Desktop\PycharmProjects\ultralytics\yolo11n.pt"
    imgsz = 640
    batch = -1

    # 执行训练
    train(yaml_path, imgsz, batch, device)
    
    # 输出训练耗时
    end_time = datetime.now()
    print(f"训练结束时间: {end_time}")
    print(f"总训练耗时: {end_time - start_time}")
