#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv26 目标检测模型训练脚本
使用 ultralytics 框架训练 bag 检测模型
"""

from ultralytics import YOLO
import os


def train_yolo_model():
    """
    训练 YOLO26n 模型
    """
    # 1. 加载预训练模型
    print("=" * 50)
    print("加载 YOLO26n 预训练模型...")
    print("=" * 50)
    model = YOLO('yolo26n.pt')
    
    # 2. 设置训练参数
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_config = os.path.join(current_dir, 'data/bag_yolo/dataset.yaml')
    
    print(f"\n数据集配置文件：{data_config}")
    print("\n开始训练模型...")
    print("=" * 50)
    
    # 3. 开始训练
    results = model.train(
        data=data_config,          # 数据集配置文件路径
        epochs=100,                # 训练轮数
        imgsz=640,                 # 输入图像尺寸
        batch=16,                  # 批次大小
        device=0,                  # 使用 GPU (0), 如使用 CPU 设置为 'cpu'
        workers=8,                 # 数据加载线程数
        optimizer='auto',          # 优化器
        lr0=0.01,                  # 初始学习率
        lrf=0.01,                  # 最终学习率 (lr0 * lrf)
        momentum=0.937,            # 动量
        weight_decay=0.0005,       # 权重衰减
        warmup_epochs=3.0,         # 预热轮数
        patience=50,               # 早停耐心值
        save=True,                 # 保存检查点
        save_period=-1,            # 每 N 个 epoch 保存一次（-1 表示只保存最后和最佳）
        verbose=True,              # 详细输出
        project='runs/detect',     # 项目目录
        name='bag_yolo26n_train',  # 实验名称
        exist_ok=False,            # 是否覆盖已有实验
        pretrained=True,           # 使用预训练权重
        amp=True,                  # 自动混合精度训练
        cache=False,               # 不缓存图像到内存
    )
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    print(f"\n训练结果保存在：runs/detect/bag_yolo26n_train/")
    print(f"最佳模型权重：runs/detect/bag_yolo26n_train/weights/best.pt")
    print(f"最后模型权重：runs/detect/bag_yolo26n_train/weights/last.pt")
    
    return results


if __name__ == '__main__':
    train_yolo_model()
