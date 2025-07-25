# -*- coding: utf-8 -*-
"""
优化的数据生成器模块
支持内存优化（实时存储）和断点续传功能
"""
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from tqdm import tqdm

from .data_generator import DataGenerator
from .data_loader import DataLoader
from .model_caller import ModelCallerFactory
from config.config import *
from config.prompt_config import prompt_manager


class OptimizedDataGenerator(DataGenerator):
    """优化的数据生成器，支持内存优化和断点续传"""
    
    def __init__(self, data_loader: DataLoader, model_caller_type: str, model_name: str, 
                 api_key: str = None, base_url: str = None):
        super().__init__(data_loader, model_caller_type, model_name, api_key, base_url)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        self.ensure_checkpoint_dir()
        self._lock = threading.Lock()
    
    def ensure_checkpoint_dir(self):
        """确保检查点目录存在"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def get_checkpoint_file(self, output_file: str, dataset_type: str) -> str:
        """获取检查点文件路径"""
        base_name = Path(output_file).stem
        checkpoint_name = f"{base_name}_{dataset_type}_checkpoint.json"
        return os.path.join(self.checkpoint_dir, checkpoint_name)
    
    def save_checkpoint(self, checkpoint_file: str, checkpoint_data: Dict):
        """保存检查点"""
        try:
            with self._lock:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存检查点失败: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """加载检查点"""
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载检查点失败: {e}")
        return None
    
    def delete_checkpoint(self, checkpoint_file: str):
        """删除检查点文件"""
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
        except Exception as e:
            print(f"删除检查点失败: {e}")
    
    def append_to_output_file(self, output_file: str, data: Dict):
        """追加数据到输出文件"""
        try:
            with self._lock:
                # 检查文件是否存在，如果不存在则创建
                if not os.path.exists(output_file):
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump([], f, ensure_ascii=False)
                
                # 读取现有数据
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # 追加新数据
                existing_data.append(data)
                
                # 写回文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"追加数据到文件失败: {e}")
    
    def generate_dataset_optimized(self, dataset_type: str, num_samples: int, output_file: str,
                                 mode: str = "complete", fixed_instruction: str = None,
                                 concurrent: bool = False, max_workers: int = 3,
                                 resume: bool = True) -> bool:
        """优化的数据集生成，支持内存优化和断点续传"""
        try:
            checkpoint_file = self.get_checkpoint_file(output_file, dataset_type)
            start_index = 0
            generated_count = 0
            
            # 检查是否需要恢复
            if resume:
                checkpoint_data = self.load_checkpoint(checkpoint_file)
                if checkpoint_data:
                    start_index = checkpoint_data.get('completed_count', 0)
                    generated_count = start_index
                    print(f"从检查点恢复，已完成 {start_index}/{num_samples} 个样本")
                    
                    # 如果已经完成，直接返回
                    if start_index >= num_samples:
                        print("数据集生成已完成")
                        self.delete_checkpoint(checkpoint_file)
                        return True
            else:
                # 不恢复时，清空输出文件和检查点
                if os.path.exists(output_file):
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump([], f, ensure_ascii=False)
                self.delete_checkpoint(checkpoint_file)
            
            # 保存初始检查点
            checkpoint_data = {
                'dataset_type': dataset_type,
                'total_samples': num_samples,
                'completed_count': generated_count,
                'mode': mode,
                'fixed_instruction': fixed_instruction,
                'concurrent': concurrent,
                'max_workers': max_workers,
                'start_time': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat(),
                'output_file': output_file
            }
            self.save_checkpoint(checkpoint_file, checkpoint_data)
            
            # 初始化进度条
            if 'streamlit' in globals() and st.session_state.get('in_streamlit', False):
                progress_bar = st.progress(generated_count / num_samples)
                status_text = st.empty()
                status_text.text(f"正在生成数据集... {generated_count}/{num_samples}")
            else:
                progress_bar = tqdm(total=num_samples, initial=generated_count, desc="生成数据集")
            
            # 生成剩余样本
            remaining_samples = num_samples - start_index
            
            if concurrent:
                success = self._generate_concurrent_optimized(
                    dataset_type, remaining_samples, output_file, checkpoint_file,
                    mode, fixed_instruction, max_workers, start_index, progress_bar
                )
            else:
                success = self._generate_sequential_optimized(
                    dataset_type, remaining_samples, output_file, checkpoint_file,
                    mode, fixed_instruction, start_index, progress_bar
                )
            
            if success:
                # 完成后删除检查点
                self.delete_checkpoint(checkpoint_file)
                print(f"数据集生成完成，共生成 {num_samples} 个样本")
            
            return success
            
        except Exception as e:
            print(f"生成数据集时发生错误: {e}")
            return False
    
    def _generate_sequential_optimized(self, dataset_type: str, num_samples: int, 
                                     output_file: str, checkpoint_file: str,
                                     mode: str, fixed_instruction: str, 
                                     start_index: int, progress_bar) -> bool:
        """串行生成（优化版本）"""
        try:
            for i in range(num_samples):
                current_index = start_index + i
                
                # 生成单个样本
                if dataset_type == "sft":
                    sample = self.sft_generator.generate_sample(mode, fixed_instruction)
                elif dataset_type == "dpo":
                    sample = self.dpo_generator.generate_sample(mode, fixed_instruction)
                else:
                    print(f"不支持的数据集类型: {dataset_type}")
                    return False
                
                if sample:
                    # 实时保存到文件
                    self.append_to_output_file(output_file, sample)
                    
                    # 更新检查点
                    checkpoint_data = self.load_checkpoint(checkpoint_file)
                    if checkpoint_data:
                        checkpoint_data['completed_count'] = current_index + 1
                        checkpoint_data['last_update'] = datetime.now().isoformat()
                        self.save_checkpoint(checkpoint_file, checkpoint_data)
                    
                    # 更新进度条
                    if hasattr(progress_bar, 'progress'):
                        # Streamlit 进度条
                        progress_bar.progress((current_index + 1) / (start_index + num_samples))
                        if 'status_text' in locals():
                            status_text.text(f"正在生成数据集... {current_index + 1}/{start_index + num_samples}")
                    else:
                        # tqdm 进度条
                        progress_bar.update(1)
                else:
                    print(f"生成第 {current_index + 1} 个样本失败")
            
            return True
            
        except Exception as e:
            print(f"串行生成过程中发生错误: {e}")
            return False
    
    def _generate_concurrent_optimized(self, dataset_type: str, num_samples: int,
                                     output_file: str, checkpoint_file: str,
                                     mode: str, fixed_instruction: str,
                                     max_workers: int, start_index: int, progress_bar) -> bool:
        """并发生成（优化版本）"""
        try:
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                futures = []
                for i in range(num_samples):
                    if dataset_type == "sft":
                        future = executor.submit(self.sft_generator.generate_sample, mode, fixed_instruction)
                    elif dataset_type == "dpo":
                        future = executor.submit(self.dpo_generator.generate_sample, mode, fixed_instruction)
                    else:
                        print(f"不支持的数据集类型: {dataset_type}")
                        return False
                    futures.append(future)
                
                # 处理完成的任务
                for future in as_completed(futures):
                    try:
                        sample = future.result()
                        if sample:
                            # 实时保存到文件
                            self.append_to_output_file(output_file, sample)
                            completed_count += 1
                            
                            # 更新检查点
                            checkpoint_data = self.load_checkpoint(checkpoint_file)
                            if checkpoint_data:
                                checkpoint_data['completed_count'] = start_index + completed_count
                                checkpoint_data['last_update'] = datetime.now().isoformat()
                                self.save_checkpoint(checkpoint_file, checkpoint_data)
                            
                            # 更新进度条
                            if hasattr(progress_bar, 'progress'):
                                # Streamlit 进度条
                                progress_bar.progress((start_index + completed_count) / (start_index + num_samples))
                                if 'status_text' in locals():
                                    status_text.text(f"正在生成数据集... {start_index + completed_count}/{start_index + num_samples}")
                            else:
                                # tqdm 进度条
                                progress_bar.update(1)
                        else:
                            print(f"生成样本失败")
                    except Exception as e:
                        print(f"处理并发任务时发生错误: {e}")
            
            return completed_count == num_samples
            
        except Exception as e:
            print(f"并发生成过程中发生错误: {e}")
            return False
    
    def get_checkpoint_status(self, output_file: str, dataset_type: str) -> Optional[Dict]:
        """获取检查点状态"""
        checkpoint_file = self.get_checkpoint_file(output_file, dataset_type)
        return self.load_checkpoint(checkpoint_file)
    
    def list_checkpoints(self) -> List[Dict]:
        """列出所有检查点"""
        checkpoints = []
        try:
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('_checkpoint.json'):
                    checkpoint_file = os.path.join(self.checkpoint_dir, file)
                    checkpoint_data = self.load_checkpoint(checkpoint_file)
                    if checkpoint_data:
                        checkpoint_data['checkpoint_file'] = file
                        checkpoints.append(checkpoint_data)
        except Exception as e:
            print(f"列出检查点时发生错误: {e}")
        return checkpoints
    
    def clean_old_checkpoints(self, days: int = 7):
        """清理旧的检查点文件"""
        try:
            current_time = time.time()
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('_checkpoint.json'):
                    file_path = os.path.join(self.checkpoint_dir, file)
                    file_time = os.path.getmtime(file_path)
                    if (current_time - file_time) > (days * 24 * 3600):
                        os.remove(file_path)
                        print(f"已删除旧检查点: {file}")
        except Exception as e:
            print(f"清理检查点时发生错误: {e}")