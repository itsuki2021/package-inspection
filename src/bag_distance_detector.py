#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
包装袋距离检测器
使用 YOLO 模型检测包装袋并判断任意两个包装袋之间的距离是否过近
"""

from ultralytics import YOLO
import cv2
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BagDetection:
    """包装袋检测结果"""
    x1: int  # 左上角 x 坐标
    y1: int  # 左上角 y 坐标
    x2: int  # 右下角 x 坐标
    y2: int  # 右下角 y 坐标
    confidence: float  # 置信度
    
    @property
    def center(self) -> Tuple[float, float]:
        """返回边界框中心点坐标"""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return (cx, cy)
    
    @property
    def width(self) -> int:
        """返回边界框宽度"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """返回边界框高度"""
        return self.y2 - self.y1


@dataclass
class BagPairDistance:
    """两个包装袋之间的距离信息"""
    bag1_idx: int
    bag2_idx: int
    distance: float  # 像素距离（边缘最短距离，如相交则为负值）
    center_distance: float  # 中心点距离
    is_too_close: bool  # 是否过近
    is_overlapping: bool  # 是否相交/重叠
    

class BagDistanceDetector:
    """包装袋距离检测器"""
    
    def __init__(self, model_path: str = 'runs/detect/bag_yolo26n_train/weights/best.pt'):
        """
        初始化检测器
        
        Args:
            model_path: YOLO 模型权重路径
        """
        print(f"加载模型：{model_path}")
        self.model = YOLO(model_path)
        print("模型加载完成")
        
    def detect_bags(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[BagDetection]:
        """
        检测图像中的所有包装袋
        
        Args:
            image: BGR 格式的图像数组
            conf_threshold: 置信度阈值
            
        Returns:
            包装袋检测结果列表
        """
        # 使用 YOLO 模型进行推理
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                detections.append(BagDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf
                ))
        
        return detections
    
    def calculate_distance(self, bag1: BagDetection, bag2: BagDetection) -> Tuple[float, float]:
        """
        计算两个包装袋之间的距离
        
        Args:
            bag1: 第一个包装袋
            bag2: 第二个包装袋
            
        Returns:
            (边缘距离，中心点距离)
            - 边缘距离：如果边界框相交则为负值（表示重叠程度），否则为正值（表示间隔）
            - 中心点距离：两个中心点之间的欧氏距离
        """
        # 计算中心点距离
        center1 = bag1.center
        center2 = bag2.center
        center_dist = math.sqrt(
            (center1[0] - center2[0]) ** 2 + 
            (center1[1] - center2[1]) ** 2
        )
        
        # 计算边界框之间的最短边缘距离
        # 这是两个矩形之间的最小距离，考虑三种情况：
        # 1. 相交（负距离）
        # 2. 水平或垂直相邻（取单一方向距离）
        # 3. 对角线分离（使用勾股定理）
        
        # 计算 x 和 y 方向的间隔/重叠
        overlap_x = min(bag1.x2, bag2.x2) - max(bag1.x1, bag2.x1)
        overlap_y = min(bag1.y2, bag2.y2) - max(bag1.y1, bag2.y1)
        
        if overlap_x > 0 and overlap_y > 0:
            # 两个方向都有重叠区域，说明两个框相交
            # 计算重叠面积的中心点到最近边的距离（作为"重叠程度"的度量）
            edge_dist = -min(overlap_x, overlap_y)
        else:
            # 至少有一个方向没有重叠，计算最短距离
            # dx: x 方向的间隔（负值表示重叠）
            # dy: y 方向的间隔（负值表示重叠）
            dx = max(bag1.x1, bag2.x1) - min(bag1.x2, bag2.x2)
            dy = max(bag1.y1, bag2.y1) - min(bag1.y2, bag2.y2)
            
            if dx <= 0 and dy <= 0:
                # 两个方向都重叠（理论上不会到这里，因为上面已经处理）
                edge_dist = max(dx, dy)
            elif dx <= 0:
                # x 方向重叠，y 方向分离
                edge_dist = dy
            elif dy <= 0:
                # y 方向重叠，x 方向分离
                edge_dist = dx
            else:
                # 两个方向都分离，计算欧氏距离
                edge_dist = math.sqrt(dx**2 + dy**2)
        
        return (round(edge_dist, 2), round(center_dist, 2))
    
    def check_bag_distances(self, detections: List[BagDetection], 
                           distance_threshold: float) -> List[BagPairDistance]:
        """
        检查所有包装袋对之间的距离
        
        Args:
            detections: 包装袋检测结果
            distance_threshold: 距离阈值（像素）
            
        Returns:
            包装袋对距离信息列表
            
        判断逻辑：
        1. 如果边界框相交（边缘距离 < 0），判定为过近
        2. 如果边缘距离 >= 0 但 < distance_threshold，判定为过近
        3. 否则为正常
        """
        pair_distances = []
        
        # 遍历所有包装袋对
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                edge_dist, center_dist = self.calculate_distance(detections[i], detections[j])
                
                # 判断是否相交（边缘距离为负）
                is_overlapping = edge_dist < 0
                
                # 判断是否过近：相交或边缘距离小于阈值
                is_too_close = is_overlapping or (edge_dist < distance_threshold)
                
                pair_distances.append(BagPairDistance(
                    bag1_idx=i,
                    bag2_idx=j,
                    distance=edge_dist,
                    center_distance=center_dist,
                    is_too_close=is_too_close,
                    is_overlapping=is_overlapping
                ))
        
        return pair_distances
    
    def process_image(self, image: np.ndarray, 
                     distance_threshold: float,
                     conf_threshold: float = 0.5) -> Tuple[List[BagDetection], List[BagPairDistance]]:
        """
        处理图像并返回检测结果和距离信息
        
        Args:
            image: BGR 格式的图像数组
            distance_threshold: 距离阈值（像素）
            conf_threshold: 置信度阈值
            
        Returns:
            (包装袋检测结果，包装袋对距离信息)
        """
        # 检测包装袋
        detections = self.detect_bags(image, conf_threshold)
        
        # 如果没有检测到包装袋，返回空列表
        if len(detections) == 0:
            return [], []
        
        # 如果只有一个包装袋，也返回空列表（没有配对）
        if len(detections) == 1:
            return detections, []
        
        # 检查所有包装袋对的距离
        pair_distances = self.check_bag_distances(detections, distance_threshold)
        
        return detections, pair_distances
    
    def draw_results(self, image: np.ndarray,
                    detections: List[BagDetection],
                    pair_distances: List[BagPairDistance],
                    distance_threshold: float) -> np.ndarray:
        """
        在图像上绘制检测结果和距离信息
        
        Args:
            image: BGR 格式的图像数组
            detections: 包装袋检测结果
            pair_distances: 包装袋对距离信息
            distance_threshold: 距离阈值（像素）
            
        Returns:
            绘制结果后的图像
        """
        result = image.copy()
        
        # 根据图像分辨率计算自适应字体大小和线宽
        height, width = image.shape[:2]
        diagonal = math.sqrt(height**2 + width**2)
        
        # 字体大小基准值（基于 640x480 标准分辨率）
        base_font_scale = 0.5 * (diagonal / 800.0)
        base_thickness = max(1, int(2 * (diagonal / 800.0)))
        
        # 边界框线宽
        box_thickness = max(2, int(3 * (diagonal / 800.0)))
        
        # 中心点半径
        center_radius = max(3, int(5 * (diagonal / 800.0)))
        
        # 连线宽度
        line_thickness = max(2, int(2 * (diagonal / 800.0)))
        
        # 统计信息字体大小
        stats_font_scale = 0.7 * (diagonal / 800.0)
        stats_thickness = max(2, int(2 * (diagonal / 800.0)))
        
        # 距离文本字体大小
        dist_font_scale = 0.5 * (diagonal / 800.0)
        dist_thickness = max(1, int(2 * (diagonal / 800.0)))
        
        # 绘制每个包装袋的边界框和编号
        for idx, detection in enumerate(detections):
            # 绘制边界框（绿色）
            cv2.rectangle(result, (detection.x1, detection.y1), 
                         (detection.x2, detection.y2), (0, 255, 0), box_thickness)
            
            # 绘制中心点
            cx, cy = map(int, detection.center)
            cv2.circle(result, (cx, cy), center_radius, (0, 0, 255), -1)
            
            # 绘制标签
            label = f'bag {idx}: {detection.confidence:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, base_thickness)[0]
            label_y = detection.y1 - 10
            
            # 确保标签不超出图像顶部
            if label_y < label_size[1]:
                label_y = label_size[1]
            
            cv2.putText(result, label, (detection.x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, (0, 255, 0), base_thickness)
        
        # 绘制包装袋之间的距离
        if len(pair_distances) > 0:
            # 找到最大的距离值用于缩放颜色
            max_dist = max(pd.center_distance for pd in pair_distances) if pair_distances else 1
            
            for pd in pair_distances:
                bag1 = detections[pd.bag1_idx]
                bag2 = detections[pd.bag2_idx]
                
                # 获取中心点
                pt1 = tuple(map(int, bag1.center))
                pt2 = tuple(map(int, bag2.center))
                
                # 根据是否相交或过近设置颜色
                if pd.is_overlapping:
                    line_color = (0, 0, 255)  # 红色 - 相交
                    text_color = (0, 0, 255)
                elif pd.is_too_close:
                    line_color = (0, 140, 255)  # 橙色 - 过近
                    text_color = (0, 140, 255)
                else:
                    # 根据距离比例设置颜色（从绿到黄）
                    ratio = min(pd.center_distance / max_dist, 1.0) if max_dist > 0 else 0
                    blue = int(255 * (1 - ratio))
                    green = 255
                    red = int(255 * ratio)
                    line_color = (blue, green, red)
                    text_color = (0, 165, 255)  # 橙色
                
                # 绘制连线
                cv2.line(result, pt1, pt2, line_color, line_thickness)
                
                # 在中点位置显示距离
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                
                # 显示边缘距离（主要）和中心距离（次要）
                if pd.is_overlapping:
                    dist_text = f'Overlap! {-pd.distance:.1f}px'
                else:
                    dist_text = f'{pd.distance:.1f}px'
                
                # 计算文本大小
                (text_w, text_h), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, dist_font_scale, dist_thickness)
                
                # 确保文本不超出图像边界
                margin = 5
                bg_x1 = max(margin, mid_x - text_w//2 - margin)
                bg_x2 = min(width - margin, mid_x + text_w//2 + margin)
                bg_y1 = max(margin, mid_y - text_h - margin)
                bg_y2 = min(height - margin, mid_y + margin)
                
                # 绘制文本背景
                cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                
                # 绘制距离文本（居中）
                text_x = max(margin, min(width - text_w - margin, mid_x - text_w//2))
                text_y = max(margin + text_h, min(height - margin, mid_y))
                cv2.putText(result, dist_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, dist_font_scale, text_color, dist_thickness)
        
        # 在图像顶部显示统计信息
        stats_y = int(30 * (diagonal / 800.0))
        cv2.putText(result, f'Total bags: {len(detections)}', (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale, (255, 255, 255), stats_thickness)
        
        # 如果有过近的情况，显示警告
        close_pairs = [pd for pd in pair_distances if pd.is_too_close]
        overlapping_pairs = [pd for pd in pair_distances if pd.is_overlapping]
        
        if len(overlapping_pairs) > 0:
            warning_text = f'CRITICAL: {len(overlapping_pairs)} pair(s) OVERLAPPING!'
            cv2.putText(result, warning_text, (10, stats_y + int(30 * (diagonal / 800.0))),
                       cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale, (0, 0, 255), stats_thickness)
        elif len(close_pairs) > 0:
            warning_text = f'WARNING: {len(close_pairs)} pair(s) too close!'
            cv2.putText(result, warning_text, (10, stats_y + int(30 * (diagonal / 800.0))),
                       cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale, (0, 140, 255), stats_thickness)
        
        # 显示距离阈值
        threshold_text = f'Threshold: {distance_threshold}px'
        cv2.putText(result, threshold_text, (10, stats_y + int(60 * (diagonal / 800.0))),
                   cv2.FONT_HERSHEY_SIMPLEX, stats_font_scale * 0.85, (200, 200, 200), stats_thickness)
        
        return result


def main():
    """测试函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法：python bag_distance_detector.py <image_path> [distance_threshold]")
        print("示例：python bag_distance_detector.py test.jpg 50")
        sys.exit(1)
    
    image_path = sys.argv[1]
    distance_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    
    # 创建检测器
    detector = BagDistanceDetector()
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像：{image_path}")
        sys.exit(1)
    
    # 处理图像
    detections, pair_distances = detector.process_image(image, distance_threshold)
    
    # 输出结果
    print(f"\n检测到 {len(detections)} 个包装袋")
    
    if len(detections) == 0:
        print("⚠️  未检测到任何包装袋")
    elif len(detections) == 1:
        print("ℹ️  只有一个包装袋，无需检测距离")
    else:
        print(f"\n共检测到 {len(pair_distances)} 对包装袋:")
        close_count = sum(1 for pd in pair_distances if pd.is_too_close)
        overlap_count = sum(1 for pd in pair_distances if pd.is_overlapping)
        
        for pd in pair_distances:
            if pd.is_overlapping:
                status = "🔴 OVERLAP!"
                dist_info = f"{-pd.distance:.2f}px overlap"
            elif pd.is_too_close:
                status = "⚠️  TOO CLOSE!"
                dist_info = f"{pd.distance:.2f}px"
            else:
                status = "✓ OK"
                dist_info = f"{pd.distance:.2f}px"
            
            print(f"  bag[{pd.bag1_idx}] <-> bag[{pd.bag2_idx}]: {dist_info} (center: {pd.center_distance:.2f}px) {status}")
        
        print(f"\n总结:")
        print(f"  - {overlap_count}/{len(pair_distances)} 对相交")
        print(f"  - {close_count}/{len(pair_distances)} 对距离过近（包括相交）")
    
    # 绘制结果
    result_image = detector.draw_results(image, detections, pair_distances, distance_threshold)
    
    # 保存结果
    output_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.jpg')
    cv2.imwrite(output_path, result_image)
    print(f"\n结果图已保存：{output_path}")


if __name__ == '__main__':
    main()
