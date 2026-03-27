# 包装袋检测系统 (Package Inspection System)

基于 YOLOv26n 目标检测技术的生产线包装袋间距检测系统，可自动检测包装袋并判断任意两个包装袋之间的距离是否过近。

## 📋 功能特性

- ✅ **自动检测**: 使用 YOLOv26n 模型自动识别图像中的所有包装袋
- 📏 **距离测量**: 计算任意两个包装袋边缘之间的最短距离（像素）
- ⚠️ **智能预警**: 当包装袋间距小于设定阈值时自动发出警告
- 🎨 **可视化展示**: 直观显示检测框、距离信息和警告标识
- 🌐 **Web 界面**: 轻量级 Web 应用，支持上传图像和实时检测

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型（可选）

如果已有训练好的模型权重（`runs/detect/*/weights/best.pt`），可跳过此步骤。

```bash
python src/train_yolo.py
```

### 3. 使用方法

#### 方法一：Web 应用（推荐）

```bash
streamlit run src/web_app.py
```

启动后在浏览器中访问，支持：
- 上传待检测图像
- 调整距离阈值参数
- 查看可视化检测结果
- 下载结果图像

#### 方法二：代码调用

```python
import cv2
from src.bag_distance_detector import BagDistanceDetector

# 初始化检测器
detector = BagDistanceDetector(model_path='yolo26n.pt')

# 读取图像
image = cv2.imread('test_image.jpg')

# 检测并处理
detections, pair_distances = detector.process_image(image, distance_threshold=50)
result_image = detector.draw_results(image, detections, pair_distances, 50)

# 保存结果
cv2.imwrite('result.jpg', result_image)
```

## 📁 项目结构

```
package-inspection/
├── data/                      # 数据集目录
│   └── bag_yolo/             # YOLO 格式训练数据
├── runs/                     # 训练输出目录（包含训练好的模型）
├── src/                      # 核心源代码
│   ├── bag_distance_detector.py  # 包装袋距离检测器
│   ├── train_yolo.py            # 模型训练脚本
│   └── web_app.py               # Web 应用
├── requirements.txt           # Python 依赖配置
└── yolo26n.pt                # YOLO 预训练模型权重
```

## 🔧 核心模块

### BagDistanceDetector 类

位于 [`src/bag_distance_detector.py`](src/bag_distance_detector.py)，提供以下功能：

- `detect_bags(image)`: 检测图像中的所有包装袋
- `process_image(image, threshold)`: 完整处理流程
- `draw_results(image, detections, pair_distances, threshold)`: 绘制可视化结果

### 距离计算说明

系统采用**边界框边缘最短距离**来判断包装袋是否过近：

- **边缘距离 < 0**: 两个包装袋相交/重叠（红色标记）
- **0 ≤ 边缘距离 < 阈值**: 包装袋过近（橙色标记）
- **边缘距离 ≥ 阈值**: 正常距离（彩色标记）

相比中心点距离，边缘距离能更准确地反映包装袋的实际间隔。

## ⚙️ 参数配置

- **距离阈值**: 默认 50 像素（可在 Web 界面或代码中调整）
- **置信度阈值**: 默认 0.5（过滤低置信度检测）

## 📊 输出说明

### 可视化结果包含
- 🟢 绿色边界框：检测到的包装袋
- 🔴 红色圆点：包装袋中心点
- 🔴 红色连线：相交/重叠的包装袋对
- 🟠 橙色连线：距离过近的包装袋对
- 🌈 彩色连线：正常距离的包装袋对
- 📝 文本标签：距离数值、统计信息

### 控制台输出
- 检测到的包装袋数量
- 每对包装袋之间的距离
- 距离过近的警告统计

## 🛠️ 开发环境

- Python >= 3.8
- Ultralytics YOLO
- OpenCV
- Streamlit
- Pillow
- NumPy

## 📝 注意事项

1. 确保已准备好训练数据集或使用预训练模型
2. 根据实际应用场景调整合适的距离阈值
3. 支持 JPG、PNG 等常见图像格式
4. 大量图像处理时建议使用 GPU 加速

## 🔮 未来扩展

- [ ] 支持视频流实时检测
- [ ] 添加更多距离度量方式
- [ ] 导出检测报告（CSV/PDF）
- [ ] API 接口服务化

---

**许可证**: 本项目仅供学习和研究使用。
