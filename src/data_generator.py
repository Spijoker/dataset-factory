# -*- coding: utf-8 -*-
"""
数据生成模块，用于生成新的数据集

注意：此模块已重构为模块化架构，推荐使用新的生成器类：
- SFTDatasetGenerator: 用于生成SFT数据集
- DPODatasetGenerator: 用于生成DPO数据集
- SFTToDPOConverter: 用于SFT到DPO的转换

此文件保留用于向后兼容性。
"""
import json
import os
import random
from typing import List, Dict, Any, Tuple, Optional
import time
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_loader import DataLoader
from src.model_caller import ModelCaller, extract_content_between_backticks

# 导入新的模块化生成器
from src.dataset_generators.sft_generator import SFTDatasetGenerator
from src.dataset_generators.dpo_generator import DPODatasetGenerator
from src.dataset_generators.sft_to_dpo_converter import SFTToDPOConverter

# 尝试导入streamlit，如果不可用则使用None
try:
    import streamlit as st
except ImportError:
    st = None


class DataGenerator:
    """
    数据生成器，用于生成新的数据集
    """
    def __init__(
        self, 
        model_caller: ModelCaller, 
        data_loader: DataLoader,
        instruction_prompt: str,
        input_prompt: str,
        output_prompt: str,
        sample_min: int = 3,
        sample_max: int = 6
    ):
        """
        初始化数据生成器
        
        Args:
            model_caller: 模型调用器
            data_loader: 数据加载器
            instruction_prompt: 生成instruction的提示模板
            input_prompt: 生成input的提示模板
            output_prompt: 生成output的提示模板
            sample_min: 最少示例数量
            sample_max: 最多示例数量
        """
        self.model_caller = model_caller
        self.data_loader = data_loader
        self.instruction_prompt = instruction_prompt
        self.input_prompt = input_prompt
        self.output_prompt = output_prompt
        self.sample_min = sample_min
        self.sample_max = sample_max
    
    def generate_instructions(self, num_to_generate: int = 1) -> List[str]:
        """
        生成新的instructions
        
        Args:
            num_to_generate: 要生成的instruction数量
            
        Returns:
            生成的instruction列表
        """
        # 获取随机示例
        examples = self.data_loader.get_random_samples(self.sample_min, self.sample_max)
        formatted_examples = self.data_loader.format_examples(examples)
        
        # 构建提示词
        prompt = self.instruction_prompt.format(
            num_to_generate=num_to_generate,
            examples=formatted_examples
        )
        
        # 调用模型生成
        response = self.model_caller.generate(prompt)
        
        # 解析生成的instructions
        instructions = []
        
        # 尝试从三个反引号中提取内容
        extracted = extract_content_between_backticks(response)
        
        if extracted:
            try:
                # 尝试将提取的内容解析为 JSON 数组
                parsed_json = json.loads(extracted)
                if isinstance(parsed_json, list):
                    # 如果是列表，则每个元素视为一个 instruction
                    instructions.extend([str(item).strip() for item in parsed_json if str(item).strip()])
                elif isinstance(parsed_json, str):
                    # 如果是字符串，按行分割
                    lines = [line.strip() for line in parsed_json.split('\n') if line.strip()]
                    instructions.extend(lines)
                else:
                    # 其他类型直接转换为字符串
                    instructions.append(str(parsed_json).strip())
            except json.JSONDecodeError:
                # 如果不是有效的 JSON，则按行分割
                lines = [line.strip() for line in extracted.split('\n') if line.strip()]
                instructions.extend(lines)
        
        # 如果提取后仍然没有足够的 instructions，或者模型直接返回了非反引号包裹的内容
        if not instructions or len(instructions) < num_to_generate:
            # 再次尝试直接处理原始响应，以防 extract_content_between_backticks 过滤掉了有效内容
            raw_lines = [line.strip() for line in response.split('\n') if line.strip()]
            for line in raw_lines:
                # 过滤掉可能的非 instruction 行（如解释性文本或多余的 'json' 标识符）
                if not line.lower().startswith('示例') and \
                   not line.lower().startswith('以下是') and \
                   not line.lower().startswith('这是') and \
                   not line.lower().startswith('json') and \
                   line not in instructions: # 避免重复添加
                    instructions.append(line)
        
        # 确保返回指定数量的instructions
        return instructions[:num_to_generate]
    
    def generate_input(self, instruction: str) -> str:
        """
        为给定的instruction生成input
        
        Args:
            instruction: 指令
            
        Returns:
            生成的input
        """
        # 获取随机示例
        examples = self.data_loader.get_random_samples(self.sample_min, self.sample_max)
        formatted_examples = self.data_loader.format_examples(examples)
        
        # 构建提示词
        prompt = self.input_prompt.format(
            instruction=instruction,
            examples=formatted_examples
        )
        
        # 调用模型生成
        response = self.model_caller.generate(prompt)
        
        # 提取生成的input
        return extract_content_between_backticks(response)
    
    def generate_output(self, instruction: str, input_text: str) -> str:
        """
        为给定的instruction和input生成output
        
        Args:
            instruction: 指令
            input_text: 输入
            
        Returns:
            生成的output
        """
        # 获取随机示例
        examples = self.data_loader.get_random_samples(self.sample_min, self.sample_max)
        formatted_examples = self.data_loader.format_examples(examples)
        
        # 构建提示词
        prompt = self.output_prompt.format(
            instruction=instruction,
            input=input_text,
            examples=formatted_examples
        )
        
        # 调用模型生成
        response = self.model_caller.generate(prompt)
        
        # 提取生成的output
        return extract_content_between_backticks(response)
    
    def generate_input_output_sample(self, instruction: str = None) -> Dict[str, str]:
        """
        为给定的instruction生成input和output（新模式）
        
        Args:
            instruction: 固定的指令，如果为None或空字符串则从原始数据集中随机选择
            
        Returns:
            包含instruction, input, output的字典
        """
        # 如果没有提供instruction或instruction为空字符串，从原始数据集中随机选择一个
        if not instruction or not instruction.strip():
            sample = self.data_loader.get_random_samples(1, 1)[0]
            instruction = sample.get('instruction', '')
            if not instruction:
                raise ValueError("原始数据集中没有找到有效的instruction")
        
        # 生成input
        input_text = self.generate_input(instruction)
        
        # 生成output
        output = self.generate_output(instruction, input_text)
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
    
    def generate_complete_sample(self) -> Dict[str, str]:
        """
        生成完整的样本（instruction, input, output）
        
        Returns:
            包含instruction, input, output的字典
        """
        # 生成instruction
        instructions = self.generate_instructions(1)
        if not instructions:
            raise ValueError("生成instruction失败")
        instruction = instructions[0]
        
        # 生成input
        input_text = self.generate_input(instruction)
        
        # 生成output
        output = self.generate_output(instruction, input_text)
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
    
    def generate_dataset(self, num_samples: int, output_file: str, mode: str = "complete", fixed_instruction: str = None, folder_mode: str = "merged", custom_filenames: Dict[str, str] = None, concurrency: int = 1) -> List[Dict[str, str]]:
        """
        生成完整的数据集
        
        Args:
            num_samples: 要生成的样本数量
            output_file: 输出文件路径
            mode: 生成模式，"complete"为完整模式，"input_output"为只生成input和output模式
            fixed_instruction: 当mode为"input_output"时使用的固定指令，如果为None则从原始数据集中随机选择
            folder_mode: 文件夹处理模式，"merged"为合并模式，"separate"为分别生成模式
            custom_filenames: 自定义文件名字典，仅在folder_mode为"separate"时使用
            concurrency: 并发请求数，默认为1（串行）
            
        Returns:
            生成的数据集
        """
        # 检查是否为文件夹输入且选择了分别生成模式
        if folder_mode == "separate" and len(self.data_loader.file_paths) > 1:
            return self.generate_dataset_for_folder_separate(num_samples, output_file, mode, fixed_instruction, custom_filenames, concurrency)
        
        # 根据并发数选择生成方式
        if concurrency > 1:
            return self._generate_dataset_concurrent(num_samples, output_file, mode, fixed_instruction, concurrency)
        else:
            return self._generate_dataset_sequential(num_samples, output_file, mode, fixed_instruction)
    
    def _generate_dataset_sequential(self, num_samples: int, output_file: str, mode: str, fixed_instruction: str = None) -> List[Dict[str, str]]:
        """
        串行生成数据集
        """
        generated_data = []
        
        # 设置进度描述
        if mode == "complete":
            desc = "生成数据集(完整模式)"
        elif fixed_instruction:
            desc = "生成数据集(固定指令模式)"
        else:
            desc = "生成数据集(随机指令模式)"
        
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
                
                # 根据模式生成样本
                if mode == "complete":
                    sample = self.generate_complete_sample()
                elif mode == "input_output":
                    sample = self.generate_input_output_sample(fixed_instruction)
                else:
                    raise ValueError(f"不支持的生成模式: {mode}")
                    
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
    
    def _generate_dataset_concurrent(self, num_samples: int, output_file: str, mode: str, fixed_instruction: str = None, concurrency: int = 3) -> List[Dict[str, str]]:
        """
        并发生成数据集，确保请求结果与内容一一对应
        """
        generated_data = [None] * num_samples  # 预分配列表，保持顺序
        
        # 设置进度描述
        if mode == "complete":
            desc = "生成数据集(并发完整模式)"
        elif fixed_instruction:
            desc = "生成数据集(并发固定指令模式)"
        else:
            desc = "生成数据集(并发随机指令模式)"
        
        # 创建Streamlit进度条（如果在Streamlit环境中）
        progress_bar = None
        status_text = None
        if st is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        completed_count = 0
        
        def generate_single_sample(index: int, generator_instance, generation_mode: str, instruction: str) -> Tuple[int, Dict[str, str]]:
            """
            生成单个样本，返回索引和样本数据以保持顺序
            """
            try:
                if generation_mode == "complete":
                    sample = generator_instance.generate_complete_sample()
                elif generation_mode == "input_output":
                    sample = generator_instance.generate_input_output_sample(instruction)
                else:
                    raise ValueError(f"不支持的生成模式: {generation_mode}")
                return index, sample
            except Exception as e:
                # 生成样本时出错
                return index, None
        
        # 使用ThreadPoolExecutor进行并发生成
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交所有任务
            future_to_index = {executor.submit(generate_single_sample, i, self, mode, fixed_instruction): i for i in range(num_samples)}
            
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
    
    def generate_dataset_for_folder_separate(self, num_samples: int, output_file: str, mode: str = "complete", fixed_instruction: str = None, custom_filenames: Dict[str, str] = None, concurrency: int = 1) -> Dict[str, Any]:
        """
        为文件夹中的每个文件分别生成数据集
        
        Args:
            num_samples: 每个文件要生成的样本数量
            output_file: 输出文件路径模板
            mode: 生成模式
            fixed_instruction: 固定指令
            custom_filenames: 自定义文件名字典，键为原文件名，值为自定义输出文件名
            concurrency: 并发请求数，默认为1（串行）
            
        Returns:
            包含所有生成结果的字典，格式为：
            {
                'all_data': [...],  # 所有数据合并
                'file_results': [   # 每个文件的结果
                    {
                        'file_name': 'xxx.json',
                        'output_path': 'path/to/output.json',
                        'data_count': 10,
                        'data': [...]
                    },
                    ...
                ]
            }
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
                    overall_status.text(f"正在处理文件 {file_idx + 1}/{total_files}: {os.path.basename(file_path)}")
                
                # 为每个文件创建单独的数据加载器
                file_loader = DataLoader(file_path)
                file_generator = DataGenerator(
                    model_caller=self.model_caller,
                    data_loader=file_loader,
                    instruction_prompt=self.instruction_prompt,
                    input_prompt=self.input_prompt,
                    output_prompt=self.output_prompt,
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
                desc = f"生成 {os.path.basename(file_path)}"
                
                # 根据并发数选择生成方式
                if concurrency > 1:
                    file_data = self._generate_file_data_concurrent(file_generator, num_samples, mode, fixed_instruction, desc, concurrency)
                else:
                    file_data = self._generate_file_data_sequential(file_generator, num_samples, mode, fixed_instruction, desc)
                
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
    
    def _generate_file_data_sequential(self, file_generator, num_samples: int, mode: str, fixed_instruction: str, desc: str) -> List[Dict[str, str]]:
        """
        为单个文件串行生成数据
        """
        file_data = []
        
        # 为单个文件创建进度条
        file_progress = None
        file_status = None
        if st is not None:
            file_progress = st.progress(0)
            file_status = st.empty()
        
        for i in tqdm(range(num_samples), desc=desc):
            try:
                # 更新文件进度条
                if file_progress is not None:
                    progress = (i + 1) / num_samples
                    file_progress.progress(progress)
                    file_status.text(f"{desc}: {i + 1}/{num_samples} ({progress:.1%})")
                
                if mode == "complete":
                    sample = file_generator.generate_complete_sample()
                elif mode == "input_output":
                    sample = file_generator.generate_input_output_sample(fixed_instruction)
                else:
                    raise ValueError(f"不支持的生成模式: {mode}")
                    
                file_data.append(sample)
                
                # 添加随机延迟
                time.sleep(random.uniform(0.5, 2.0))
                
            except Exception as e:
                # 生成样本时出错
                continue
        
        # 完成文件进度条
        if file_progress is not None:
            file_progress.progress(1.0)
            file_status.text(f"{desc}: 完成 ({len(file_data)}/{num_samples})")
        
        return file_data
    
    def _generate_file_data_concurrent(self, file_generator, num_samples: int, mode: str, fixed_instruction: str, desc: str, concurrency: int) -> List[Dict[str, str]]:
        """
        为单个文件并发生成数据，确保请求结果与内容一一对应
        """
        file_data = [None] * num_samples  # 预分配列表，保持顺序
        
        # 为单个文件创建进度条
        file_progress = None
        file_status = None
        if st is not None:
            file_progress = st.progress(0)
            file_status = st.empty()
        
        completed_count = 0
        
        def generate_single_sample(index: int, generator_instance, generation_mode: str, instruction: str) -> Tuple[int, Dict[str, str]]:
            """
            生成单个样本，返回索引和样本数据以保持顺序
            """
            try:
                if generation_mode == "complete":
                    sample = generator_instance.generate_complete_sample()
                elif generation_mode == "input_output":
                    sample = generator_instance.generate_input_output_sample(instruction)
                else:
                    raise ValueError(f"不支持的生成模式: {generation_mode}")
                return index, sample
            except Exception as e:
                # 生成样本时出错
                return index, None
        
        # 使用ThreadPoolExecutor进行并发生成
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交所有任务
            future_to_index = {executor.submit(generate_single_sample, i, file_generator, mode, fixed_instruction): i for i in range(num_samples)}
            
            # 处理完成的任务
            for future in as_completed(future_to_index):
                index, sample = future.result()
                if sample is not None:
                    file_data[index] = sample
                
                completed_count += 1
                
                # 更新进度条
                if file_progress is not None:
                    progress = completed_count / num_samples
                    file_progress.progress(progress)
                    file_status.text(f"{desc}: {completed_count}/{num_samples} ({progress:.1%})")
        
        # 过滤掉None值，保持原有顺序
        final_data = [sample for sample in file_data if sample is not None]
        
        # 完成进度条
        if file_progress is not None:
            file_progress.progress(1.0)
            file_status.text(f"{desc}: 完成 ({len(final_data)}/{num_samples})")
        
        return final_data
    
    def _get_short_timestamp(self) -> str:
        """
        生成短时间戳，格式为MMDD_HHMM
        
        Returns:
            短时间戳字符串
        """
        import datetime
        now = datetime.datetime.now()
        return now.strftime("%m%d_%H%M")