#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
包装袋检测 Web 应用
支持用户上传图像、设置距离阈值、显示检测结果
"""

import streamlit as st
from bag_distance_detector import BagDistanceDetector, BagDetection, BagPairDistance
import cv2
import numpy as np
from PIL import Image
import tempfile
import os


def convert_to_opencv(image: Image.Image) -> np.ndarray:
    """将 PIL 图像转换为 OpenCV 格式（BGR）"""
    img_array = np.array(image.convert('RGB'))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def draw_results_on_image(detector: BagDistanceDetector,
                         image: np.ndarray,
                         detections: list,
                         pair_distances: list,
                         distance_threshold: float) -> np.ndarray:
    """使用检测器绘制结果"""
    return detector.draw_results(image, detections, pair_distances, distance_threshold)


def main():
    st.set_page_config(
        page_title="包装袋间距检测系统",
        page_icon="📦",
        layout="wide"
    )
    
    # 自定义 CSS 样式
    st.markdown("""
        <style>
        .stAlert {
            margin-top: 1rem;
        }
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 标题
    st.markdown('<h1 class="main-header">📦 包装袋间距检测系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于 YOLO26n 目标检测技术</p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 参数设置")
        
        distance_threshold = st.slider(
            "距离阈值（像素）",
            min_value=10,
            max_value=1000,
            value=50,
            step=10,
            help="两个包装袋中心点之间的距离小于此值时，判定为过近"
        )
        
        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="检测框的置信度低于此值将被过滤"
        )
        
        st.divider()
        
        st.info("""
        **使用说明：**
        1. 上传包含包装袋的图像
        2. 调整距离阈值参数
        3. 查看检测结果和警告信息
        4. 下载结果图像
        """)
    
    # 主界面布局
    col1, col2 = st.columns(2)
    
    # 初始化检测器（使用缓存避免重复加载）
    @st.cache_resource
    def load_detector():
        try:
            # 尝试使用训练好的模型
            model_path = 'runs/detect/bag_yolo26n_train/weights/best.pt'
            if not os.path.exists(model_path):
                # 如果训练未完成，使用预训练模型
                model_path = 'yolo26n.pt'
            return BagDistanceDetector(model_path)
        except Exception as e:
            st.error(f"模型加载失败：{str(e)}")
            return None
    
    detector = load_detector()
    
    # 文件上传区域
    with col1:
        st.header("📤 上传图像")
        
        uploaded_file = st.file_uploader(
            "选择一张图像进行检测",
            type=['jpg', 'jpeg', 'png'],
            help="支持 JPG、JPEG、PNG 格式"
        )
        
        if uploaded_file is not None:
            # 显示原始图像
            image = Image.open(uploaded_file)
            st.image(image, caption="原始图像", use_container_width=True)
            
            # 保存临时文件用于处理
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # 处理按钮
            process_btn = st.button("🔍 开始检测", type="primary", use_container_width=True)
            
            if process_btn and detector is not None:
                try:
                    with st.spinner("正在分析图像..."):
                        # 转换为 OpenCV 格式
                        cv_image = convert_to_opencv(image)
                        
                        # 执行检测
                        detections, pair_distances = detector.process_image(
                            cv_image, 
                            distance_threshold,
                            conf_threshold
                        )
                        
                        # 绘制结果
                        result_image = draw_results_on_image(
                            detector,
                            cv_image,
                            detections,
                            pair_distances,
                            distance_threshold
                        )
                        
                        # 转换回 RGB 用于显示
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        
                        st.session_state['result_image'] = result_image_rgb
                        st.session_state['detections'] = detections
                        st.session_state['pair_distances'] = pair_distances
                        st.session_state['cv_image'] = cv_image
                        
                except Exception as e:
                    st.error(f"检测过程中出现错误：{str(e)}")
                    st.exception(e)
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    # 结果显示区域
    with col2:
        st.header("📊 检测结果")
        
        if 'result_image' in st.session_state:
            # 显示结果图像
            st.image(
                st.session_state['result_image'],
                caption="检测结果图",
                use_container_width=True
            )
            
            # 统计信息
            detections = st.session_state['detections']
            pair_distances = st.session_state['pair_distances']
            
            st.divider()
            
            # 基本统计
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("包装袋数量", len(detections))
            
            with col_stat2:
                total_pairs = len(pair_distances)
                st.metric("检测对数", total_pairs)
            
            with col_stat3:
                close_pairs = sum(1 for pd in pair_distances if pd.is_too_close)
                overlapping = sum(1 for pd in pair_distances if pd.is_overlapping)
                st.metric("过近对数", f"{close_pairs}\n({overlapping}相交)")
            
            st.divider()
            
            # 详细结果
            if len(detections) == 0:
                st.warning("⚠️ 未检测到任何包装袋", icon="⚠️")
            elif len(detections) == 1:
                st.info("ℹ️ 只检测到一个包装袋，无需进行距离检测", icon="ℹ️")
            else:
                # 检查是否有过近或相交的情况
                overlapping_pairs = [pd for pd in pair_distances if pd.is_overlapping]
                close_pairs_list = [pd for pd in pair_distances if pd.is_too_close and not pd.is_overlapping]
                
                if len(overlapping_pairs) > 0:
                    st.error(
                        f"🚨 **严重**: 发现 {len(overlapping_pairs)} 对包装袋相互重叠！",
                        icon="🚨"
                    )
                    
                    # 显示相交的详细信息
                    with st.expander("查看相交的包装袋详情", expanded=True):
                        for idx, pd in enumerate(overlapping_pairs, 1):
                            st.write(f"**第 {idx} 对相交**:")
                            st.write(f"- 包装袋 [{pd.bag1_idx}] ↔ 包装袋 [{pd.bag2_idx}]")
                            st.write(f"- 重叠程度：**{-pd.distance:.1f} 像素**")
                            st.write(f"- 中心距离：{pd.center_distance:.1f} 像素")
                
                if len(close_pairs_list) > 0:
                    st.warning(
                        f"⚠️ **警告**: 发现 {len(close_pairs_list)} 对包装袋距离过近！",
                        icon="⚠️"
                    )
                    
                    # 显示过近的详细信息
                    with st.expander("查看过近的包装袋详情", expanded=len(overlapping_pairs)==0):
                        for idx, pd in enumerate(close_pairs_list, 1):
                            st.write(f"**第 {idx} 对过近**:")
                            st.write(f"- 包装袋 [{pd.bag1_idx}] ↔ 包装袋 [{pd.bag2_idx}]")
                            st.write(f"- 边缘距离：**{pd.distance:.1f} 像素**")
                            st.write(f"- 中心距离：{pd.center_distance:.1f} 像素")
                            st.write(f"- 阈值：{distance_threshold} 像素")
                
                if len(overlapping_pairs) == 0 and len(close_pairs_list) == 0:
                    st.success(
                        f"✅ 所有包装袋间距正常（阈值：{distance_threshold} 像素）",
                        icon="✅"
                    )
                
                # 显示所有距离详情
                with st.expander("查看所有包装袋对的距离详情"):
                    for pd in pair_distances:
                        if pd.is_overlapping:
                            status = "🔴 相交"
                            info = f"重叠 {-pd.distance:.1f}px (中心：{pd.center_distance:.1f}px)"
                        elif pd.is_too_close:
                            status = "⚠️ 过近"
                            info = f"{pd.distance:.1f}px (中心：{pd.center_distance:.1f}px)"
                        else:
                            status = "✅ 正常"
                            info = f"{pd.distance:.1f}px (中心：{pd.center_distance:.1f}px)"
                        
                        st.write(
                            f"bag[{pd.bag1_idx}] ↔ bag[{pd.bag2_idx}]: {info} {status}"
                        )
            
            # 下载按钮
            st.divider()
            
            # 使用绘制好结果的图像用于下载（而不是原始图像）
            result_image_rgb = st.session_state['result_image']
            result_pil = Image.fromarray(result_image_rgb)
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='_result.png') as tmp_file:
                result_pil.save(tmp_file, format='PNG')
                tmp_result_path = tmp_file.name
            
            with open(tmp_result_path, 'rb') as file:
                st.download_button(
                    label="📥 下载结果图像",
                    data=file.read(),
                    file_name="detection_result.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            # 清理临时文件
            if os.path.exists(tmp_result_path):
                os.unlink(tmp_result_path)
        
        else:
            # 默认提示信息
            st.info(
                """
                👈 请从左侧上传图像
                
                **功能特点：**
                - 🎯 自动检测所有包装袋位置
                - 📏 计算任意两个包装袋之间的距离
                - ⚠️ 智能判断距离是否过近
                - 📊 可视化展示检测结果
                - 💾 支持下载结果图像
                """
            )
    
    # 底部说明
    st.divider()
    st.caption(
        "💡 **提示**: 🔴 红色连线表示相交（重叠），🟠 橙色连线表示距离过近，彩色连线表示正常距离（颜色越红表示距离越远）。"
        "绿色方框为检测到的包装袋，红色圆点为中心点。"
        "距离显示的是边界框边缘之间的最短距离（负值表示相交）。"
    )


if __name__ == "__main__":
    main()
