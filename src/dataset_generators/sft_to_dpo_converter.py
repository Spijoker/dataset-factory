# -*- coding: utf-8 -*-
"""
SFT到DPO数据集转换器
用于将现有的SFT数据集转换为DPO格式，为每条数据自动生成rejected字段
"""

import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_generator import BaseDatasetGenerator
from ..data_loader import DataLoader
from ..model_caller import ModelCaller, extract_content_between_backticks

# 尝试导入streamlit，如果不可用则使用None
try:
    import streamlit as st
except ImportError:
    st = None


class SFTToDPOConverter(BaseDatasetGenerator):
    """
    SFT到DPO数据集转换器
    
    将现有的SFT数据集（包含instruction、input、output字段）转换为DPO格式：
    - instruction: 保持不变
    - input: 保持不变
    - chosen: 原来的output字段
    - rejected: 新生成的劣质回答
    """
    
    def __init__(
        self,
        model_caller: ModelCaller,
        data_loader: DataLoader,
        rejected_prompt: str,
        sample_min: int = 3,
        sample_max: int = 6
    ):
        """
        初始化SFT到DPO转换器
        
        Args:
            model_caller: 模型调用器
            data_loader: 数据加载器（加载SFT数据集）
            rejected_prompt: 生成rejected（劣质回答）的提示模板
            sample_min: 最少示例数量
            sample_max: 最多示例数量
        """
        super().__init__(model_caller, data_loader, sample_min, sample_max)
        self.rejected_prompt = rejected_prompt
    
    def get_dataset_format_description(self) -> str:
        """
        获取数据集格式描述
        """
        return "SFT转DPO"
    
    def generate_sample(self, **kwargs) -> Dict[str, str]:
        """
        转换单个SFT样本为DPO格式
        
        Returns:
            包含instruction, input, chosen, rejected的字典
        """
        # 从原始数据集中随机选择一个样本
        sft_sample = self.get_random_examples()[0]
        return self.convert_sft_sample_to_dpo(sft_sample)
    
    def convert_sft_sample_to_dpo(self, sft_sample: Dict[str, Any]) -> Dict[str, str]:
        """
        将单个SFT样本转换为DPO格式
        
        Args:
            sft_sample: SFT样本，包含instruction、input、output字段
            
        Returns:
            DPO格式的样本，包含instruction、input、chosen、rejected字段
        """
        instruction = sft_sample.get('instruction', '')
        input_text = sft_sample.get('input', '')
        chosen = sft_sample.get('output', '')  # 原来的output作为chosen
        
        if not instruction:
            raise ValueError("SFT样本中缺少instruction字段")
        if not chosen:
            raise ValueError("SFT样本中缺少output字段")
        
        # 生成rejected（劣质回答）
        rejected = self.generate_rejected(instruction, input_text, chosen)
        
        return {
            "instruction": instruction,
            "input": input_text,
            "chosen": chosen,
            "rejected": rejected
        }
    
    def generate_rejected(self, instruction: str, input_text: str, chosen: str) -> str:
        """
        为给定的instruction、input和chosen生成rejected（劣质回答）
        
        Args:
            instruction: 指令
            input_text: 输入
            chosen: 优质回答（原SFT数据集的output）
            
        Returns:
            生成的rejected（劣质回答）
        """
        # 获取随机示例（用于提供上下文）
        examples = self.get_random_examples()
        formatted_examples = self.format_examples(examples)
        
        # 构建提示词
        prompt = self.rejected_prompt.format(
            instruction=instruction,
            input=input_text,
            chosen=chosen,
            examples=formatted_examples
        )
        
        # 调用模型生成
        response = self.model_caller.generate(prompt)
        
        # 提取生成的rejected
        return extract_content_between_backticks(response)
    
    def convert_sft_dataset_to_dpo(
        self,
        sft_file_path: str,
        output_file: str,
        concurrency: int = 1
    ) -> List[Dict[str, str]]:
        """
        将整个SFT数据集转换为DPO格式
        
        Args:
            sft_file_path: SFT数据集文件路径
            output_file: 输出DPO数据集文件路径
            concurrency: 并发请求数，默认为1（串行）
            
        Returns:
            转换后的DPO数据集
        """
        # 加载SFT数据集
        with open(sft_file_path, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)
        
        if not isinstance(sft_data, list):
            raise ValueError("SFT数据集必须是JSON数组格式")
        
        # 开始转换SFT数据集
        
        # 根据并发数选择转换方式
        if concurrency > 1:
            dpo_data = self._convert_concurrent(sft_data, concurrency)
        else:
            dpo_data = self._convert_sequential(sft_data, output_file)
        
        # 保存转换后的数据
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        
        # 转换完成
        
        return dpo_data
    
    def _convert_sequential(self, sft_data: List[Dict[str, Any]], output_file: str = None) -> List[Dict[str, str]]:
        """
        串行转换SFT数据集
        """
        dpo_data = []
        
        # 创建Streamlit进度条（如果在Streamlit环境中）
        progress_bar = None
        status_text = None
        if st is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # 使用tqdm显示终端进度
        for i, sft_sample in enumerate(tqdm(sft_data, desc="转换SFT到DPO")):
            try:
                # 更新Streamlit进度条
                if progress_bar is not None:
                    progress = (i + 1) / len(sft_data)
                    progress_bar.progress(progress)
                    status_text.text(f"转换SFT到DPO: {i + 1}/{len(sft_data)} ({progress:.1%})")
                
                # 转换单个样本
                dpo_sample = self.convert_sft_sample_to_dpo(sft_sample)
                dpo_data.append(dpo_sample)
                
                # 每转换10个样本保存一次，防止中途失败
                if (i + 1) % 10 == 0 and output_file:
                    temp_output = f"{os.path.splitext(output_file)[0]}_temp.json"
                    with open(temp_output, 'w', encoding='utf-8') as f:
                        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                # 转换样本时出错
                continue
        
        # 完成进度条
        if progress_bar is not None:
            progress_bar.progress(1.0)
            status_text.text(f"转换SFT到DPO: 完成 ({len(dpo_data)}/{len(sft_data)})")
        
        return dpo_data
    
    def _convert_concurrent(
        self,
        sft_data: List[Dict[str, Any]],
        concurrency: int = 3
    ) -> List[Dict[str, str]]:
        """
        并发转换SFT数据集
        """
        dpo_data = [None] * len(sft_data)  # 预分配列表，保持顺序
        
        # 创建Streamlit进度条（如果在Streamlit环境中）
        progress_bar = None
        status_text = None
        if st is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        completed_count = 0
        
        def convert_single_sample(index: int, sft_sample: Dict[str, Any]) -> tuple:
            """
            转换单个样本，返回索引和转换后的数据以保持顺序
            """
            try:
                dpo_sample = self.convert_sft_sample_to_dpo(sft_sample)
                return index, dpo_sample
            except Exception as e:
                # 转换样本时出错
                return index, None
        
        # 使用ThreadPoolExecutor进行并发转换
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(convert_single_sample, i, sft_sample): i 
                for i, sft_sample in enumerate(sft_data)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_index):
                index, dpo_sample = future.result()
                if dpo_sample is not None:
                    dpo_data[index] = dpo_sample
                
                completed_count += 1
                
                # 更新进度条
                if progress_bar is not None:
                    progress = completed_count / len(sft_data)
                    progress_bar.progress(progress)
                    status_text.text(f"并发转换SFT到DPO: {completed_count}/{len(sft_data)} ({progress:.1%})")
        
        # 过滤掉None值，保持原有顺序
        final_data = [sample for sample in dpo_data if sample is not None]
        
        # 完成进度条
        if progress_bar is not None:
            progress_bar.progress(1.0)
            status_text.text(f"并发转换SFT到DPO: 完成 ({len(final_data)}/{len(sft_data)})")
        
        return final_data
    
    def convert_folder_sft_to_dpo(
        self,
        sft_folder_path: str,
        output_folder: str,
        concurrency: int = 1
    ) -> Dict[str, Any]:
        """
        批量转换文件夹中的所有SFT数据集为DPO格式
        
        Args:
            sft_folder_path: SFT数据集文件夹路径
            output_folder: 输出文件夹路径
            concurrency: 并发请求数
            
        Returns:
            转换结果统计
        """
        # 获取文件夹中的所有JSON文件
        sft_files = []
        for file_name in os.listdir(sft_folder_path):
            if file_name.endswith('.json'):
                sft_files.append(os.path.join(sft_folder_path, file_name))
        
        if not sft_files:
            raise ValueError(f"在文件夹 {sft_folder_path} 中没有找到JSON文件")
        
        # 找到SFT数据集文件
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        conversion_results = []
        total_converted = 0
        
        for sft_file in sft_files:
            try:
                file_name = os.path.basename(sft_file)
                output_file = os.path.join(output_folder, f"dpo_{file_name}")
                
                # 正在转换文件
                
                # 转换单个文件
                dpo_data = self.convert_sft_dataset_to_dpo(
                    sft_file, output_file, concurrency
                )
                
                conversion_results.append({
                    'input_file': sft_file,
                    'output_file': output_file,
                    'converted_count': len(dpo_data)
                })
                
                total_converted += len(dpo_data)
                
            except Exception as e:
                # 转换文件时出错
                continue
        
        # 批量转换完成
        # 成功转换文件
        
        return {
            'total_files': len(sft_files),
            'converted_files': len(conversion_results),
            'total_converted_samples': total_converted,
            'results': conversion_results
        }