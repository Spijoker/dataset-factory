# -*- coding: utf-8 -*-
"""
基础数据集生成器
定义数据集生成的通用接口和共享功能
"""

import json
import os
import random
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data_loader import DataLoader
from ..model_caller import ModelCaller

# 尝试导入streamlit，如果不可用则使用None
try:
    import streamlit as st
except ImportError:
    st = None


class BaseDatasetGenerator(ABC):
    """
    数据集生成器基类
    定义通用接口和共享功能
    """
    
    def __init__(
        self,
        model_caller: ModelCaller,
        data_loader: DataLoader,
        sample_min: int = 3,
        sample_max: int = 6
    ):
        """
        初始化基础数据集生成器
        
        Args:
            model_caller: 模型调用器
            data_loader: 数据加载器
            sample_min: 最少示例数量
            sample_max: 最多示例数量
        """
        self.model_caller = model_caller
        self.data_loader = data_loader
        self.sample_min = sample_min
        self.sample_max = sample_max
    
    @abstractmethod
    def generate_sample(self, **kwargs) -> Dict[str, Any]:
        """
        生成单个样本
        
        Returns:
            生成的样本数据
        """
        pass
    
    @abstractmethod
    def get_dataset_format_description(self) -> str:
        """
        获取数据集格式描述
        
        Returns:
            数据集格式的描述字符串
        """
        pass
    
    def generate_dataset(
        self,
        num_samples: int,
        output_file: str,
        folder_mode: str = "merged",
        custom_filenames: Optional[Dict[str, str]] = None,
        concurrency: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        生成完整的数据集
        
        Args:
            num_samples: 要生成的样本数量
            output_file: 输出文件路径
            folder_mode: 文件夹处理模式，"merged"为合并模式，"separate"为分别生成模式
            custom_filenames: 自定义文件名字典，仅在folder_mode为"separate"时使用
            concurrency: 并发请求数，默认为1（串行）
            **kwargs: 其他参数
            
        Returns:
            生成的数据集
        """
        # 检查是否为文件夹输入且选择了分别生成模式
        if folder_mode == "separate" and len(self.data_loader.file_paths) > 1:
            return self._generate_dataset_for_folder_separate(
                num_samples, output_file, custom_filenames, concurrency, **kwargs
            )
        
        # 根据并发数选择生成方式
        if concurrency > 1:
            return self._generate_dataset_concurrent(
                num_samples, output_file, concurrency, **kwargs
            )
        else:
            return self._generate_dataset_sequential(
                num_samples, output_file, **kwargs
            )
    
    def _generate_dataset_sequential(
        self,
        num_samples: int,
        output_file: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        串行生成数据集
        """
        generated_data = []
        
        # 设置进度描述
        desc = f"生成{self.get_dataset_format_description()}数据集"
        
        # 创建Streamlit进度条（如果在Streamlit环境中）
        progress_bar = None
        status_text = None
        if st is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # 使用tqdm显示终端进度
        for i in tqdm(range(num_samples), desc=desc):
            try:
                # 更新Streamlit进度条
                if progress_bar is not None:
                    progress = (i + 1) / num_samples
                    progress_bar.progress(progress)
                    status_text.text(f"{desc}: {i + 1}/{num_samples} ({progress:.1%})")
                
                # 生成样本
                sample = self.generate_sample(**kwargs)
                generated_data.append(sample)
                
                # 每生成10个样本保存一次，防止中途失败
                if (i + 1) % 10 == 0:
                    self.data_loader.save_data(generated_data, output_file)
                
                # 添加随机延迟，避免频繁请求
                time.sleep(random.uniform(0.5, 2.0))
                
            except Exception as e:
                # 生成样本时出错
                continue
        
        # 完成进度条
        if progress_bar is not None:
            progress_bar.progress(1.0)
            status_text.text(f"{desc}: 完成 ({len(generated_data)}/{num_samples})")
        
        # 最终保存
        self.data_loader.save_data(generated_data, output_file)
        
        return generated_data
    
    def _generate_dataset_concurrent(
        self,
        num_samples: int,
        output_file: str,
        concurrency: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        并发生成数据集，确保请求结果与内容一一对应
        """
        generated_data = [None] * num_samples  # 预分配列表，保持顺序
        
        # 设置进度描述
        desc = f"并发生成{self.get_dataset_format_description()}数据集"
        
        # 创建Streamlit进度条（如果在Streamlit环境中）
        progress_bar = None
        status_text = None
        if st is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        completed_count = 0
        
        def generate_single_sample(index: int) -> tuple:
            """
            生成单个样本，返回索引和样本数据以保持顺序
            """
            try:
                sample = self.generate_sample(**kwargs)
                return index, sample
            except Exception as e:
                # 生成样本时出错
                return index, None
        
        # 使用ThreadPoolExecutor进行并发生成
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(generate_single_sample, i): i 
                for i in range(num_samples)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_index):
                index, sample = future.result()
                if sample is not None:
                    generated_data[index] = sample
                
                completed_count += 1
                
                # 更新进度条
                if progress_bar is not None:
                    progress = completed_count / num_samples
                    progress_bar.progress(progress)
                    status_text.text(f"{desc}: {completed_count}/{num_samples} ({progress:.1%})")
                
                # 每完成10个样本保存一次（只保存非None的样本）
                if completed_count % 10 == 0:
                    valid_data = [sample for sample in generated_data if sample is not None]
                    if valid_data:
                        self.data_loader.save_data(valid_data, output_file)
        
        # 过滤掉None值，保持原有顺序
        final_data = [sample for sample in generated_data if sample is not None]
        
        # 完成进度条
        if progress_bar is not None:
            progress_bar.progress(1.0)
            status_text.text(f"{desc}: 完成 ({len(final_data)}/{num_samples})")
        
        # 最终保存
        if final_data:
            self.data_loader.save_data(final_data, output_file)
        
        return final_data
    
    def _generate_dataset_for_folder_separate(
        self,
        num_samples: int,
        output_file: str,
        custom_filenames: Optional[Dict[str, str]] = None,
        concurrency: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        为文件夹中的每个文件分别生成数据集
        
        Args:
            num_samples: 每个文件要生成的样本数量
            output_file: 输出文件路径模板
            custom_filenames: 自定义文件名字典
            concurrency: 并发请求数
            **kwargs: 其他参数
            
        Returns:
            包含所有生成结果的字典
        """
        all_generated_data = []
        file_results = []
        output_dir = os.path.dirname(output_file)
        
        # 创建Streamlit进度条（如果在Streamlit环境中）
        overall_progress = None
        overall_status = None
        if st is not None:
            overall_progress = st.progress(0)
            overall_status = st.empty()
        
        total_files = len(self.data_loader.file_paths)
        
        for file_idx, file_path in enumerate(self.data_loader.file_paths):
            try:
                # 更新总体进度
                if overall_progress is not None:
                    progress = file_idx / total_files
                    overall_progress.progress(progress)
                    overall_status.text(
                        f"正在处理文件 {file_idx + 1}/{total_files}: {os.path.basename(file_path)}"
                    )
                
                # 为每个文件创建单独的数据加载器和生成器
                file_loader = DataLoader(file_path)
                file_generator = self.__class__(
                    model_caller=self.model_caller,
                    data_loader=file_loader,
                    sample_min=self.sample_min,
                    sample_max=self.sample_max
                )
                
                # 生成文件名
                original_filename = os.path.basename(file_path)
                
                if custom_filenames and original_filename in custom_filenames:
                    # 使用自定义文件名
                    output_filename = custom_filenames[original_filename]
                    if not output_filename.endswith('.json'):
                        output_filename += '.json'
                else:
                    # 使用默认命名规则
                    file_base_name = os.path.splitext(original_filename)[0]
                    timestamp = self._get_short_timestamp()
                    output_filename = f"{file_base_name}_{timestamp}.json"
                
                file_output_path = os.path.join(output_dir, output_filename)
                
                # 正在为文件生成数据集
                
                # 生成数据
                if concurrency > 1:
                    file_data = file_generator._generate_dataset_concurrent(
                        num_samples, file_output_path, concurrency, **kwargs
                    )
                else:
                    file_data = file_generator._generate_dataset_sequential(
                        num_samples, file_output_path, **kwargs
                    )
                
                # 保存单个文件的结果
                if file_data:
                    file_loader.save_data(file_data, file_output_path)
                    all_generated_data.extend(file_data)
                    
                    # 记录文件结果
                    file_results.append({
                        'file_name': os.path.basename(file_path),
                        'output_path': file_output_path,
                        'data_count': len(file_data),
                        'data': file_data
                    })
                    
                    # 为文件生成了样本
                
            except Exception as e:
                # 处理文件时出错
                continue
        
        # 完成总体进度
        if overall_progress is not None:
            overall_progress.progress(1.0)
            overall_status.text(f"所有文件处理完成: {len(file_results)}/{total_files} 个文件")
        
        return {
            'all_data': all_generated_data,
            'file_results': file_results
        }
    
    def _get_short_timestamp(self) -> str:
        """
        获取短时间戳
        
        Returns:
            格式为MMDD_HHMM的时间戳
        """
        return time.strftime("%m%d_%H%M")
    
    def get_random_examples(self) -> List[Dict[str, Any]]:
        """
        获取随机示例
        
        Returns:
            随机示例列表
        """
        return self.data_loader.get_random_samples(self.sample_min, self.sample_max)
    
    def format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """
        格式化示例
        
        Args:
            examples: 示例列表
            
        Returns:
            格式化后的示例字符串
        """
        return self.data_loader.format_examples(examples)