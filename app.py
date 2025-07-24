#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集生成器可视化界面
使用Streamlit构建的现代化Web界面
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# 导入配置和模块
import config
from src.data_loader import DataLoader
from src.model_caller import ModelCallerFactory
from src.data_generator import DataGenerator
from src.dataset_generators.sft_generator import SFTDatasetGenerator
from src.dataset_generators.dpo_generator import DPODatasetGenerator
from src.dataset_generators.sft_to_dpo_converter import SFTToDPOConverter
from src.utils import setup_directories, get_timestamp, analyze_dataset, split_dataset
from src.prompt_config import prompt_manager

# 页面配置
st.set_page_config(
    page_title="数据集生成器",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """初始化session state"""
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = []
    if 'generation_complete' not in st.session_state:
        st.session_state.generation_complete = False
    if 'current_stats' not in st.session_state:
        st.session_state.current_stats = None
    if 'file_results' not in st.session_state:
        st.session_state.file_results = None
    if 'selected_input_path' not in st.session_state:
        st.session_state.selected_input_path = None
    if 'selected_output_path' not in st.session_state:
        st.session_state.selected_output_path = config.OUTPUT_DIR
    if 'preview_file_path' not in st.session_state:
        st.session_state.preview_file_path = None
    if 'dataset_type' not in st.session_state:
        st.session_state.dataset_type = "SFT"
    if 'conversion_results' not in st.session_state:
        st.session_state.conversion_results = None

def select_folder():
    """使用tkinter文件对话框选择文件夹"""
    def folder_dialog():
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 置顶显示
        
        selected_path = filedialog.askdirectory(
            title="选择数据集文件夹",
            parent=root
        )
        
        root.destroy()
        return selected_path
    
    return folder_dialog()

def select_file():
    """使用tkinter文件对话框选择文件"""
    def file_dialog():
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 置顶显示
        
        selected_path = filedialog.askopenfilename(
            title="选择数据集文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            parent=root
        )
        
        root.destroy()
        return selected_path
    
    return file_dialog()

def select_output_folder():
    """使用tkinter文件对话框选择输出文件夹"""
    def folder_dialog():
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        root.attributes('-topmost', True)  # 置顶显示
        
        selected_path = filedialog.askdirectory(
            title="选择输出文件夹",
            parent=root
        )
        
        root.destroy()
        return selected_path
    
    return folder_dialog()



def display_dataset_preview(input_path, max_samples=3):
    """显示数据集预览（支持文件和文件夹）"""
    try:
        if os.path.isdir(input_path):
            # 文件夹模式：显示文件列表和选择功能
            temp_loader = DataLoader(input_path)
            
            st.write(f"📁 **文件夹信息**: {len(temp_loader.file_paths)} 个JSON文件")
            st.write(f"📊 **总数据量**: 共 {len(temp_loader.data)} 条数据")
            
            # 显示可点击的文件列表
            st.write("**点击文件名查看预览**:")
            
            # 创建文件选择按钮
            cols = st.columns(min(3, len(temp_loader.file_paths)))  # 最多3列
            for i, file_path in enumerate(temp_loader.file_paths):
                col_index = i % len(cols)
                with cols[col_index]:
                    file_name = os.path.basename(file_path)
                    # 检查是否是当前选择的文件
                    is_selected = st.session_state.preview_file_path == file_path
                    button_type = "primary" if is_selected else "secondary"
                    
                    if st.button(
                        f"📄 {file_name}", 
                        key=f"preview_file_{i}",
                        type=button_type,
                        use_container_width=True
                    ):
                        st.session_state.preview_file_path = file_path
                        st.rerun()
            
            # 显示选中文件的预览
            if st.session_state.preview_file_path and st.session_state.preview_file_path in temp_loader.file_paths:
                st.write(f"\n**当前预览**: {os.path.basename(st.session_state.preview_file_path)}")
                
                # 加载单个文件的数据
                single_file_loader = DataLoader(st.session_state.preview_file_path)
                file_data = single_file_loader.data
                
                st.write(f"📊 **文件数据量**: 共 {len(file_data)} 条数据")
                
                if file_data:
                    preview_data = file_data[:max_samples]
                    
                    for i, item in enumerate(preview_data):
                        with st.expander(f"样本 {i+1}", expanded=(i==0)):
                            st.write(f"**Instruction**: {item.get('instruction', 'N/A')}")
                            st.write(f"**Input**: {item.get('input', 'N/A')}")
                            st.write(f"**Output**: {item.get('output', 'N/A')}")
            else:
                st.info("👆 点击上方文件名查看具体文件的数据预览")
                
        else:
            # 单文件模式：直接显示预览
            temp_loader = DataLoader(input_path)
            data = temp_loader.data
            
            st.write(f"📊 **数据集信息**: 共 {len(data)} 条数据")
            
            if data:
                st.write("**数据预览**:")
                preview_data = data[:max_samples]
                
                for i, item in enumerate(preview_data):
                    with st.expander(f"样本 {i+1}", expanded=(i==0)):
                        st.write(f"**Instruction**: {item.get('instruction', 'N/A')}")
                        st.write(f"**Input**: {item.get('input', 'N/A')}")
                        st.write(f"**Output**: {item.get('output', 'N/A')}")
                        
    except Exception as e:
        st.error(f"无法加载数据集预览: {str(e)}")

def main():
    """主函数"""
    init_session_state()
    
    # 主标题
    st.markdown('<h1 class="main-header">🤖 Dataset-Factory</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于AI的高质量训练数据生成工具</p>', unsafe_allow_html=True)
    
    # 主导航标签页
    tab1, tab2 = st.tabs(["📊 数据集生成", "⚙️ 提示词配置"])
    
    with tab1:
        show_dataset_generation()
    
    with tab2:
        show_prompt_config()
    
def show_prompt_config():
    """显示提示词配置界面"""
    st.markdown("## ⚙️ 提示词配置管理")
    st.markdown("在这里可以查看和编辑用于生成数据集的提示词模板")
    
    # 提示词类型选择
    prompt_type = st.selectbox(
        "选择提示词类型",
        ["SFT生成", "DPO生成", "SFT转DPO"],
        help="选择要配置的提示词类型"
    )
    
    if prompt_type == "SFT生成":
        show_sft_prompts()
    elif prompt_type == "DPO生成":
        show_dpo_prompts()
    else:
        show_sft_to_dpo_prompts()

def show_sft_prompts():
    """显示SFT提示词配置"""
    st.markdown("### 📝 SFT数据集生成提示词")
    
    # 获取当前提示词
    current_prompts = prompt_manager.get_sft_prompts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 指令生成提示词")
        instruction_prompt = st.text_area(
            "Instruction生成提示词",
            value=current_prompts['instruction'],
            height=200,
            help="用于生成instruction字段的提示词"
        )
        
        st.markdown("#### 输入生成提示词")
        input_prompt = st.text_area(
            "Input生成提示词",
            value=current_prompts['input'],
            height=200,
            help="用于生成input字段的提示词"
        )
    
    with col2:
        st.markdown("#### 输出生成提示词")
        output_prompt = st.text_area(
            "Output生成提示词",
            value=current_prompts['output'],
            height=200,
            help="用于生成output字段的提示词"
        )
        
        st.markdown("#### 操作")
        col_save, col_reset = st.columns(2)
        
        with col_save:
            if st.button("💾 保存配置", type="primary", use_container_width=True):
                try:
                    prompt_manager.update_sft_prompts(
                        instruction_prompt=instruction_prompt,
                        input_prompt=input_prompt,
                        output_prompt=output_prompt
                    )
                    st.success("✅ SFT提示词配置已保存！")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 保存失败: {str(e)}")
        
        with col_reset:
            if st.button("🔄 重置为默认", use_container_width=True):
                try:
                    prompt_manager.reset_sft_prompts()
                    st.success("✅ 已重置为默认配置！")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 重置失败: {str(e)}")
    
    # 预览区域
    st.markdown("### 👀 提示词预览")
    with st.expander("查看完整提示词", expanded=False):
        st.markdown("**Instruction生成提示词:**")
        st.code(instruction_prompt, language="text")
        st.markdown("**Input生成提示词:**")
        st.code(input_prompt, language="text")
        st.markdown("**Output生成提示词:**")
        st.code(output_prompt, language="text")

def show_dpo_prompts():
    """显示DPO提示词配置"""
    st.markdown("### 📝 DPO数据集生成提示词")
    
    # 获取当前提示词
    current_prompts = prompt_manager.get_dpo_prompts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 指令生成提示词")
        instruction_prompt = st.text_area(
            "Instruction生成提示词",
            value=current_prompts['instruction'],
            height=150,
            help="用于生成instruction字段的提示词"
        )
        
        st.markdown("#### 输入生成提示词")
        input_prompt = st.text_area(
            "Input生成提示词",
            value=current_prompts['input'],
            height=150,
            help="用于生成input字段的提示词"
        )
    
    with col2:
        st.markdown("#### 优质回答生成提示词")
        chosen_prompt = st.text_area(
            "Chosen生成提示词",
            value=current_prompts['chosen'],
            height=150,
            help="用于生成chosen字段的提示词"
        )
        
        st.markdown("#### 劣质回答生成提示词")
        rejected_prompt = st.text_area(
            "Rejected生成提示词",
            value=current_prompts['rejected'],
            height=150,
            help="用于生成rejected字段的提示词"
        )
    
    # 操作按钮
    col_save, col_reset = st.columns(2)
    
    with col_save:
        if st.button("💾 保存配置", type="primary", use_container_width=True, key="save_dpo"):
            try:
                prompt_manager.update_dpo_prompts(
                    instruction_prompt=instruction_prompt,
                    input_prompt=input_prompt,
                    chosen_prompt=chosen_prompt,
                    rejected_prompt=rejected_prompt
                )
                st.success("✅ DPO提示词配置已保存！")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 保存失败: {str(e)}")
    
    with col_reset:
        if st.button("🔄 重置为默认", use_container_width=True, key="reset_dpo"):
            try:
                prompt_manager.reset_dpo_prompts()
                st.success("✅ 已重置为默认配置！")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 重置失败: {str(e)}")
    
    # 预览区域
    st.markdown("### 👀 提示词预览")
    with st.expander("查看完整提示词", expanded=False):
        st.markdown("**Instruction生成提示词:**")
        st.code(instruction_prompt, language="text")
        st.markdown("**Input生成提示词:**")
        st.code(input_prompt, language="text")
        st.markdown("**Chosen生成提示词:**")
        st.code(chosen_prompt, language="text")
        st.markdown("**Rejected生成提示词:**")
        st.code(rejected_prompt, language="text")

def show_sft_to_dpo_prompts():
    """显示SFT转DPO提示词配置"""
    st.markdown("### 📝 SFT转DPO提示词")
    
    # 获取当前提示词
    current_prompts = prompt_manager.get_sft_to_dpo_prompts()
    
    st.markdown("#### 劣质回答生成提示词")
    st.markdown("用于为现有SFT数据集生成rejected字段的提示词")
    
    rejected_prompt = st.text_area(
        "Rejected生成提示词",
        value=current_prompts['rejected'],
        height=300,
        help="用于生成rejected字段的提示词，将基于现有的instruction和input生成劣质回答"
    )
    
    # 操作按钮
    col_save, col_reset = st.columns(2)
    
    with col_save:
        if st.button("💾 保存配置", type="primary", use_container_width=True, key="save_sft_to_dpo"):
            try:
                prompt_manager.update_sft_to_dpo_prompts(
                    rejected_prompt=rejected_prompt
                )
                st.success("✅ SFT转DPO提示词配置已保存！")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 保存失败: {str(e)}")
    
    with col_reset:
        if st.button("🔄 重置为默认", use_container_width=True, key="reset_sft_to_dpo"):
            try:
                prompt_manager.reset_sft_to_dpo_prompts()
                st.success("✅ 已重置为默认配置！")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 重置失败: {str(e)}")
    
    # 预览区域
    st.markdown("### 👀 提示词预览")
    with st.expander("查看完整提示词", expanded=False):
        st.markdown("**Rejected生成提示词:**")
        st.code(rejected_prompt, language="text")
    
    # 使用说明
    st.markdown("### 📖 使用说明")
    st.info("""
    **SFT转DPO模式说明:**
    
    - 此模式用于将现有的SFT数据集转换为DPO格式
    - 系统会保留原有的instruction、input和output字段
    - output字段会被重命名为chosen（优质回答）
    - 使用配置的提示词为每条数据生成rejected字段（劣质回答）
    - 最终生成包含instruction、input、chosen、rejected四个字段的DPO数据集
    """)

def show_dataset_generation():
    """显示数据集生成界面"""
    # 侧边栏配置
    with st.sidebar:
        st.markdown("## ⚙️ 配置选项")
        
        # 数据集类型选择
        st.markdown("### 🎯 数据集类型")
        dataset_type = st.selectbox(
            "选择数据集类型",
            ["SFT", "DPO", "SFT转DPO"],
            index=["SFT", "DPO", "SFT转DPO"].index(st.session_state.dataset_type),
            help="SFT: 监督微调数据集\nDPO: 直接偏好优化数据集\nSFT转DPO: 将现有SFT数据集转换为DPO格式"
        )
        st.session_state.dataset_type = dataset_type
        
        # 输入设置
        st.markdown("### 📁 输入设置")
        
        # 浏览选择文件或文件夹
        st.markdown("**浏览选择文件或文件夹**")
        
        if st.session_state.selected_input_path:
            st.text_input(
                "已选择的路径",
                value=st.session_state.selected_input_path,
                disabled=True,
                key="selected_path_display"
            )
        else:
            st.info("点击下方按钮选择文件或文件夹")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 选择文件夹", use_container_width=True, key="browse_folder"):
                try:
                    selected_path = select_folder()
                    if selected_path:
                        st.session_state.selected_input_path = selected_path
                        st.rerun()
                except Exception as e:
                    st.error(f"选择文件夹时出错: {str(e)}")
        
        with col2:
            if st.button("📄 选择文件", use_container_width=True, key="browse_file"):
                try:
                    selected_path = select_file()
                    if selected_path:
                        st.session_state.selected_input_path = selected_path
                        st.rerun()
                except Exception as e:
                    st.error(f"选择文件时出错: {str(e)}")
        
        # 确定最终选择的数据集
        current_dataset = None
        if st.session_state.selected_input_path:
            current_dataset = st.session_state.selected_input_path
            st.success(f"✅ 当前使用: {os.path.basename(current_dataset)}")
        else:
            st.warning("⚠️ 请选择一个数据集文件或文件夹")
        
        # 检查数据集是否发生变化，如果变化则清除预览文件选择
        if 'last_selected_dataset' not in st.session_state:
            st.session_state.last_selected_dataset = None
        
        if st.session_state.last_selected_dataset != current_dataset:
            st.session_state.preview_file_path = None
            st.session_state.last_selected_dataset = current_dataset
        
        selected_dataset = current_dataset
        
        # 清除选择按钮
        if st.session_state.selected_input_path:
            if st.button("🗑️ 清除选择", key="clear_input"):
                st.session_state.selected_input_path = None
                st.rerun()
        
        # 输出设置
        st.markdown("### 📤 输出设置")
        
        # 输出文件夹选择
        col1, col2 = st.columns([3, 1])
        
        with col1:
            output_folder = st.text_input(
                "输出文件夹",
                value=st.session_state.selected_output_path,
                help="生成的数据集将保存到此文件夹",
                key="output_folder_input"
            )
            # 更新session state
            st.session_state.selected_output_path = output_folder
        
        with col2:
            if st.button("📁 浏览", use_container_width=True, key="browse_output"):
                try:
                    selected_path = select_output_folder()
                    if selected_path:
                        st.session_state.selected_output_path = selected_path
                        st.rerun()
                except Exception as e:
                    st.error(f"选择文件夹时出错: {str(e)}")
        
        output_filename = st.text_input(
            "输出文件名（可选）",
            placeholder="留空将自动生成带时间戳的文件名",
            help="不需要包含.json扩展名"
        )
        
        # 模型配置
        st.markdown("### 🧠 模型设置")
        model_type = st.selectbox(
            "模型类型",
            ["ollama", "openai_compatible"],
            index=0 if config.MODEL_TYPE == "ollama" else 1
        )
        
        model_name = st.text_input(
            "模型名称",
            value=config.MODEL_NAME,
            help="例如: deepseek-r1:8b, gpt-3.5-turbo"
        )
        
        # 并发数设置
        concurrency = st.number_input(
            "并发请求数",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="同时发送的请求数量，建议根据模型服务器性能调整。过高可能导致请求失败或被限流。"
        )
        
        if model_type == "openai_compatible":
            api_key = st.text_input(
                "API Key",
                value=config.OPENAI_API_KEY or "",
                type="password"
            )
            base_url = st.text_input(
                "Base URL",
                value=config.OPENAI_BASE_URL or "",
                help="API的基础URL"
            )
        else:
            api_key = None
            base_url = None
        
        # 生成模式
        st.markdown("### 🎯 生成模式")
        
        if st.session_state.dataset_type == "SFT":
            generation_mode = st.radio(
                "选择生成模式",
                ["完整模式", "Input/Output模式"],
                help="完整模式：生成instruction+input+output\nInput/Output模式：只生成input+output，instruction可固定或从原数据集随机选择"
            )
        elif st.session_state.dataset_type == "DPO":
            generation_mode = st.radio(
                "选择生成模式",
                ["完整模式", "固定指令模式"],
                help="完整模式：生成instruction+input+chosen+rejected\n固定指令模式：使用固定instruction，生成input+chosen+rejected"
            )
        else:  # SFT转DPO
            st.info("💡 SFT转DPO模式：为现有SFT数据集的每条数据自动生成rejected字段")
            generation_mode = "转换模式"
        
        fixed_instruction = None
        if (st.session_state.dataset_type == "SFT" and generation_mode == "Input/Output模式") or \
           (st.session_state.dataset_type == "DPO" and generation_mode == "固定指令模式"):
            fixed_instruction = st.text_area(
                "固定指令（可选）" if st.session_state.dataset_type == "SFT" else "固定指令",
                height=100,
                help="留空则从原始数据集中随机选择instruction；填写则所有生成的数据都使用这个固定的instruction" if st.session_state.dataset_type == "SFT" else "所有生成的数据都使用这个固定的instruction",
                placeholder="留空将从原数据集中随机选择instruction..." if st.session_state.dataset_type == "SFT" else "请输入固定的instruction..."
            )
            # 如果用户输入了空白字符，将其转换为None
            if fixed_instruction and not fixed_instruction.strip():
                fixed_instruction = None
        
        # 文件夹处理模式（仅在选择文件夹时显示）
        folder_mode = "merged"
        custom_filenames = {}
        if selected_dataset and os.path.isdir(selected_dataset):
            st.markdown("### 📁 文件夹处理模式")
            folder_mode = st.radio(
                "选择文件夹处理方式",
                ["合并生成", "分别生成"],
                help="合并生成：将文件夹中所有文件的内容合并后生成一个数据集\n分别生成：为文件夹中每个文件单独生成数据集",
                key="folder_mode_radio"
            )
            folder_mode = "merged" if folder_mode == "合并生成" else "separate"
            
            # 如果选择分别生成，显示文件名设置选项
            if folder_mode == "separate":
                st.markdown("#### 📝 输出文件名设置")
                
                # 获取文件夹中的JSON文件列表
                try:
                    temp_loader = DataLoader(selected_dataset)
                    json_files = [os.path.basename(fp) for fp in temp_loader.file_paths]
                    
                    use_custom_names = st.checkbox(
                        "为每个文件自定义输出文件名",
                        help="勾选后可以为每个文件单独设置输出文件名，否则使用默认命名规则"
                    )
                    
                    if use_custom_names:
                        st.write("为以下文件设置输出文件名：")
                        for i, json_file in enumerate(json_files):
                            file_base_name = os.path.splitext(json_file)[0]
                            default_name = f"dataset_{file_base_name}"
                            
                            custom_name = st.text_input(
                                f"📄 {json_file}",
                                value=default_name,
                                key=f"custom_filename_{i}",
                                help="输出文件名（不需要包含.json扩展名）"
                            )
                            
                            if custom_name:
                                if not custom_name.endswith('.json'):
                                    custom_name += '.json'
                                custom_filenames[json_file] = custom_name
                            else:
                                custom_filenames[json_file] = f"{default_name}.json"
                    
                except Exception as e:
                    st.warning(f"无法读取文件夹内容: {str(e)}")
        
        # 生成参数
        st.markdown("### 📊 生成参数")
        num_samples = st.number_input(
            "生成样本数量",
            min_value=1,
            max_value=1000,
            value=config.GENERATION_NUM,
            step=1
        )
        
        sample_min = st.number_input(
            "最少示例数量",
            min_value=1,
            max_value=10,
            value=config.SAMPLE_MIN,
            step=1
        )
        
        sample_max = st.number_input(
            "最多示例数量",
            min_value=sample_min,
            max_value=20,
            value=config.SAMPLE_MAX,
            step=1
        )
        
        # 其他选项
        st.markdown("### 🔧 其他选项")
        enable_analysis = st.checkbox("生成后分析数据集", value=True)
        enable_split = st.checkbox("分割训练集和验证集", value=False)
        
        if enable_split:
            train_ratio = st.slider(
                "训练集比例",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05
            )
        else:
            train_ratio = 0.8
    
    # 主内容区域
    if selected_dataset:
        # 数据集预览
        st.markdown("## 📋 数据集预览")
        with st.expander("查看数据集详情", expanded=False):
            display_dataset_preview(selected_dataset)
        
        # 生成控制
        st.markdown("## 🚀 开始生成")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            generate_button = st.button(
                "🎯 开始生成数据集",
                type="primary",
                use_container_width=True
            )
            
            # 显示当前模式信息
            if generation_mode == "Input/Output模式":
                if fixed_instruction:
                    st.info(f"💡 使用固定指令: {fixed_instruction[:50]}{'...' if len(fixed_instruction) > 50 else ''}")
                else:
                    st.info("💡 将从原数据集中随机选择instruction")
        
        with col2:
            if st.session_state.generated_data:
                if st.button("📥 下载结果", use_container_width=True):
                    # 创建下载链接
                    json_str = json.dumps(st.session_state.generated_data, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="💾 下载JSON文件",
                        data=json_str,
                        file_name=f"generated_dataset_{get_timestamp()}.json",
                        mime="application/json",
                        use_container_width=True
                    )
        
        with col3:
            if st.button("🗑️ 清空结果", use_container_width=True):
                st.session_state.generated_data = []
                st.session_state.generation_complete = False
                st.session_state.current_stats = None
                st.session_state.file_results = None
                st.rerun()
        
        # 生成过程
        if generate_button:
            try:
                # 确保目录存在
                setup_directories([config.DATA_DIR, config.INPUT_DIR, output_folder])
                os.makedirs(output_folder, exist_ok=True)
                
                # 初始化组件
                with st.spinner("🔄 正在初始化..."):
                    data_loader = DataLoader(selected_dataset)
                    model_caller = ModelCallerFactory.create(
                        model_type=model_type,
                        model_name=model_name,
                        api_key=api_key,
                        base_url=base_url
                    )
                    
                    # 获取当前提示词配置
                    if st.session_state.dataset_type == "SFT":
                        prompts = prompt_manager.get_sft_prompts()
                        data_generator = SFTDatasetGenerator(
                            model_caller=model_caller,
                            data_loader=data_loader,
                            instruction_prompt=prompts['instruction'],
                            input_prompt=prompts['input'],
                            output_prompt=prompts['output'],
                            sample_min=sample_min,
                            sample_max=sample_max
                        )
                    elif st.session_state.dataset_type == "DPO":
                        prompts = prompt_manager.get_dpo_prompts()
                        data_generator = DPODatasetGenerator(
                            model_caller=model_caller,
                            data_loader=data_loader,
                            instruction_prompt=prompts['instruction'],
                            input_prompt=prompts['input'],
                            chosen_prompt=prompts['chosen'],
                            rejected_prompt=prompts['rejected'],
                            sample_min=sample_min,
                            sample_max=sample_max
                        )
                    else:  # SFT转DPO
                        prompts = prompt_manager.get_sft_to_dpo_prompts()
                        data_generator = SFTToDPOConverter(
                            model_caller=model_caller,
                            data_loader=data_loader,
                            rejected_prompt=prompts['rejected'],
                            sample_min=sample_min,
                            sample_max=sample_max
                        )
                
                # 生成数据集
                st.markdown("### 📈 生成进度")
                
                # 根据数据集类型和生成模式确定mode参数
                if st.session_state.dataset_type == "SFT":
                    mode = "complete" if generation_mode == "完整模式" else "input_output"
                elif st.session_state.dataset_type == "DPO":
                    mode = "complete" if generation_mode == "完整模式" else "input_output"  # DPO的固定指令模式对应input_output
                else:  # SFT转DPO
                    mode = "convert"  # 转换模式
                
                # 生成输出文件名
                if output_filename:
                    # 用户指定了文件名
                    if not output_filename.endswith('.json'):
                        output_filename += '.json'
                    output_file = os.path.join(output_folder, output_filename)
                else:
                    # 自动生成文件名，使用原数据集名称作为前缀
                    if os.path.isdir(selected_dataset):
                        # 文件夹模式
                        dataset_name = os.path.basename(selected_dataset.rstrip(os.sep))
                    else:
                        # 单文件模式
                        dataset_name = os.path.splitext(os.path.basename(selected_dataset))[0]
                    
                    # 使用短时间戳
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
                    output_file = os.path.join(output_folder, f"{dataset_name}_{timestamp}.json")
                
                # 根据数据集类型调用不同的方法
                with st.spinner("🔄 正在生成数据集..."):
                    if st.session_state.dataset_type == "SFT转DPO":
                        # SFT转DPO使用转换方法
                        if os.path.isdir(selected_dataset) and folder_mode == "separate":
                            result = data_generator.convert_folder_sft_to_dpo(
                                sft_folder_path=selected_dataset,
                                output_folder=output_folder,
                                concurrency=concurrency
                            )
                        else:
                            result = data_generator.convert_sft_dataset_to_dpo(
                                sft_file_path=selected_dataset,
                                output_file=output_file,
                                concurrency=concurrency
                            )
                    else:
                        # SFT和DPO使用生成方法
                        result = data_generator.generate_dataset(
                            num_samples=num_samples,
                            output_file=output_file,
                            mode=mode,
                            fixed_instruction=fixed_instruction,
                            folder_mode=folder_mode,
                            custom_filenames=custom_filenames if folder_mode == "separate" else None,
                            concurrency=concurrency
                        )
                
                # 处理不同的返回格式
                if isinstance(result, dict) and 'all_data' in result:
                    # 多文件分别生成模式
                    generated_data = result['all_data']
                    st.session_state.generated_data = generated_data
                    st.session_state.file_results = result['file_results']
                    st.session_state.generation_complete = True
                    
                    st.info(f"💾 已为文件夹中的每个文件分别生成数据集到: {output_folder}")
                    
                    # 显示每个文件的生成结果
                    st.markdown("### 📁 各文件生成结果")
                    for file_result in result['file_results']:
                        st.write(f"- **{file_result['file_name']}**: {file_result['data_count']} 个样本 → `{os.path.basename(file_result['output_path'])}`")
                    
                    st.success(f"🎉 总共成功生成 {len(generated_data)} 个样本！")
                else:
                    # 单文件或合并模式
                    generated_data = result
                    st.session_state.generated_data = generated_data
                    st.session_state.file_results = None
                    st.session_state.generation_complete = True
                    
                    st.info(f"💾 文件已保存到: {output_file}")
                    st.success(f"🎉 成功生成 {len(generated_data)} 个样本！")
                
                # 分析数据集
                if enable_analysis and generated_data:
                    with st.spinner("📊 正在分析数据集..."):
                        stats = analyze_dataset(output_file)
                        st.session_state.current_stats = stats
                
                # 分割数据集
                if enable_split and generated_data:
                    with st.spinner("✂️ 正在分割数据集..."):
                        split_files = split_dataset(output_file, train_ratio)
                        st.info(f"📁 训练集: {os.path.basename(split_files['train'])}")
                        st.info(f"📁 验证集: {os.path.basename(split_files['val'])}")
                
            except Exception as e:
                st.error(f"❌ 生成过程中出现错误: {str(e)}")
        
        # 显示结果
        if st.session_state.generated_data:
            st.markdown("## 📊 生成结果")
            
            # 统计信息
            if st.session_state.current_stats:
                stats = st.session_state.current_stats
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总样本数", stats['total_samples'])
                with col2:
                    st.metric("平均指令长度", f"{stats['instruction_length']['avg']:.1f}")
                with col3:
                    st.metric("平均输入长度", f"{stats['input_length']['avg']:.1f}")
                with col4:
                    st.metric("平均输出长度", f"{stats['output_length']['avg']:.1f}")
            
            # 数据预览
            st.markdown("### 📋 数据预览")
            
            # 转换为DataFrame用于显示
            df = pd.DataFrame(st.session_state.generated_data)
            
            # 显示表格
            st.dataframe(
                df,
                use_container_width=True,
                height=400
            )
            
            # 多文件结果查看（如果有）
            if st.session_state.file_results:
                st.markdown("### 📁 分文件查看")
                
                # 文件选择
                file_options = [f"{result['file_name']} ({result['data_count']} 样本)" for result in st.session_state.file_results]
                selected_file_idx = st.selectbox(
                    "选择要查看的文件",
                    range(len(file_options)),
                    format_func=lambda x: file_options[x]
                )
                
                selected_file_result = st.session_state.file_results[selected_file_idx]
                
                # 显示文件信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("文件名", selected_file_result['file_name'])
                with col2:
                    st.metric("样本数量", selected_file_result['data_count'])
                with col3:
                    st.metric("输出文件", os.path.basename(selected_file_result['output_path']))
                
                # 显示该文件的数据
                if selected_file_result['data']:
                    st.markdown("#### 📋 文件数据预览")
                    file_df = pd.DataFrame(selected_file_result['data'])
                    st.dataframe(file_df, use_container_width=True, height=300)
                    
                    # 文件内样本详细查看
                    st.markdown("#### 🔍 样本详细查看")
                    sample_index = st.selectbox(
                        "选择样本",
                        range(len(selected_file_result['data'])),
                        format_func=lambda x: f"样本 {x+1}",
                        key=f"file_sample_{selected_file_idx}"
                    )
                    
                    sample = selected_file_result['data'][sample_index]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Instruction:**")
                        st.text_area(
                            "指令",
                            value=sample.get('instruction', ''),
                            height=100,
                            disabled=True,
                            key=f"file_inst_{selected_file_idx}_{sample_index}"
                        )
                        
                        st.markdown("**Input:**")
                        st.text_area(
                            "输入",
                            value=sample.get('input', ''),
                            height=150,
                            disabled=True,
                            key=f"file_input_{selected_file_idx}_{sample_index}"
                        )
                    
                    with col2:
                        # 根据数据集类型显示不同字段
                        if st.session_state.dataset_type in ["DPO", "SFT转DPO"]:
                            st.markdown("**Chosen:**")
                            st.text_area(
                                "优质回答",
                                value=sample.get('chosen', ''),
                                height=120,
                                disabled=True,
                                key=f"file_chosen_{selected_file_idx}_{sample_index}"
                            )
                            
                            st.markdown("**Rejected:**")
                            st.text_area(
                                "劣质回答",
                                value=sample.get('rejected', ''),
                                height=120,
                                disabled=True,
                                key=f"file_rejected_{selected_file_idx}_{sample_index}"
                            )
                        else:
                            st.markdown("**Output:**")
                            st.text_area(
                                "输出",
                                value=sample.get('output', ''),
                                height=255,
                                disabled=True,
                                key=f"file_output_{selected_file_idx}_{sample_index}"
                            )
            
            # 详细查看（全部数据）
            st.markdown("### 🔍 全部数据详细查看")
            if len(st.session_state.generated_data) > 0:
                sample_index = st.selectbox(
                    "选择样本",
                    range(len(st.session_state.generated_data)),
                    format_func=lambda x: f"样本 {x+1}"
                )
                
                sample = st.session_state.generated_data[sample_index]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Instruction:**")
                    st.text_area(
                        "指令",
                        value=sample.get('instruction', ''),
                        height=100,
                        disabled=True,
                        key=f"inst_{sample_index}"
                    )
                    
                    st.markdown("**Input:**")
                    st.text_area(
                        "输入",
                        value=sample.get('input', ''),
                        height=150,
                        disabled=True,
                        key=f"input_{sample_index}"
                    )
                
                with col2:
                    # 根据数据集类型显示不同字段
                    if st.session_state.dataset_type in ["DPO", "SFT转DPO"]:
                        st.markdown("**Chosen:**")
                        st.text_area(
                            "优质回答",
                            value=sample.get('chosen', ''),
                            height=120,
                            disabled=True,
                            key=f"chosen_{sample_index}"
                        )
                        
                        st.markdown("**Rejected:**")
                        st.text_area(
                            "劣质回答",
                            value=sample.get('rejected', ''),
                            height=120,
                            disabled=True,
                            key=f"rejected_{sample_index}"
                        )
                    else:
                        st.markdown("**Output:**")
                        st.text_area(
                            "输出",
                            value=sample.get('output', ''),
                            height=255,
                            disabled=True,
                            key=f"output_{sample_index}"
                        )
    
    else:
        st.warning("⚠️ 请先在侧边栏选择一个数据集文件")
        
        # 显示帮助信息
        st.markdown("## 📖 使用说明")
        
        with st.expander("🚀 快速开始", expanded=True):
            st.markdown("""
            1. **准备数据集**: 将JSON格式的数据集文件放在项目根目录或`data/input/`目录下
            2. **选择数据集**: 在左侧边栏选择要使用的数据集文件
            3. **配置模型**: 选择模型类型和名称
            4. **选择数据集类型**: 
               - **SFT**: 监督微调数据集
               - **DPO**: 直接偏好优化数据集
               - **SFT转DPO**: 将现有SFT数据集转换为DPO格式
            5. **选择生成模式**: 
               - **完整模式**: 生成全新的instruction、input和output/chosen+rejected
               - **Input/Output模式**: 使用固定的instruction，只生成input和output
               - **固定指令模式**: 使用固定instruction，生成input+chosen+rejected（DPO）
               - **转换模式**: 为SFT数据集自动生成rejected字段（SFT转DPO）
            5. **设置参数**: 调整生成数量和其他参数
            6. **开始生成**: 点击"开始生成数据集"按钮
            7. **查看结果**: 生成完成后可以预览、分析和下载结果
            """)
        
        with st.expander("💡 功能特性"):
            st.markdown("""
            - 🎯 **多类型数据集**: 支持SFT、DPO和SFT转DPO三种数据集类型
            - 🔄 **智能转换**: 自动将SFT数据集转换为DPO格式，生成rejected字段
            - 🧠 **多模型支持**: 兼容Ollama和OpenAI兼容的API
            - 📊 **实时预览**: 生成过程中实时显示进度和结果
            - 📈 **数据分析**: 自动分析生成数据的统计信息
            - ✂️ **数据分割**: 自动分割训练集和验证集
            - 💾 **便捷下载**: 一键下载生成的数据集
            - 🎨 **美观界面**: 现代化的Web界面设计
            - 📁 **批量处理**: 支持文件夹批量生成和转换
            """)

if __name__ == "__main__":
    main()