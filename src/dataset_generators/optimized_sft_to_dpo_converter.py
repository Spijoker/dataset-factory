# -*- coding: utf-8 -*-
"""
优化的SFT到DPO数据集转换器
支持断点续传和内存优化功能
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .sft_to_dpo_converter import SFTToDPOConverter
from ..data_loader import DataLoader
from ..model_caller import ModelCaller

# 尝试导入streamlit，如果不可用则使用None
try:
    import streamlit as st
except ImportError:
    st = None


class OptimizedSFTToDPOConverter(SFTToDPOConverter):
    """
    优化的SFT到DPO数据集转换器
    
    在原有转换器基础上增加：
    - 断点续传功能
    - 内存优化（实时保存）
    - 更详细的进度跟踪
    - 错误恢复机制
    """
    
    def __init__(
        self,
        model_caller: ModelCaller,
        data_loader: DataLoader,
        rejected_prompt: str,
        checkpoint_dir: str = "checkpoints",
        sample_min: int = 3,
        sample_max: int = 6
    ):
        """
        初始化优化的SFT到DPO转换器
        
        Args:
            model_caller: 模型调用器
            data_loader: 数据加载器
            rejected_prompt: 生成rejected的提示模板
            checkpoint_dir: 检查点保存目录
            sample_min: 最少示例数量
            sample_max: 最多示例数量
        """
        super().__init__(model_caller, data_loader, rejected_prompt, sample_min, sample_max)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _get_checkpoint_path(self, output_file: str) -> str:
        """获取检查点文件路径"""
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        return os.path.join(self.checkpoint_dir, f"{base_name}_checkpoint.json")
    
    def _save_checkpoint(self, checkpoint_path: str, data: Dict[str, Any]):
        """保存检查点"""
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载检查点失败: {e}")
        return None
    
    def _delete_checkpoint(self, checkpoint_path: str):
        """删除检查点文件"""
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except Exception as e:
                print(f"删除检查点失败: {e}")
    
    def convert_sft_dataset_to_dpo_optimized(
        self,
        sft_file_path: str,
        output_file: str,
        concurrency: int = 1,
        resume_conversion: bool = True,
        save_interval: int = 5
    ) -> List[Dict[str, str]]:
        """
        优化版本的SFT到DPO转换，支持断点续传和内存优化
        
        Args:
            sft_file_path: SFT数据集文件路径
            output_file: 输出DPO数据集文件路径
            concurrency: 并发请求数
            resume_conversion: 是否启用断点续传
            save_interval: 保存间隔（每转换多少个样本保存一次）
            
        Returns:
            转换后的DPO数据集
        """
        # 加载SFT数据集
        with open(sft_file_path, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)
        
        if not isinstance(sft_data, list):
            raise ValueError("SFT数据集必须是JSON数组格式")
        
        checkpoint_path = self._get_checkpoint_path(output_file)
        start_index = 0
        dpo_data = []
        
        # 尝试从检查点恢复
        if resume_conversion:
            checkpoint = self._load_checkpoint(checkpoint_path)
            if checkpoint:
                start_index = checkpoint.get('completed_count', 0)
                dpo_data = checkpoint.get('converted_data', [])
                
                if st is not None:
                    st.info(f"🔄 从检查点恢复转换，已完成 {start_index}/{len(sft_data)} 个样本")
                print(f"从检查点恢复转换，已完成 {start_index}/{len(sft_data)} 个样本")
        
        # 如果已经全部完成，直接返回
        if start_index >= len(sft_data):
            if st is not None:
                st.success("✅ 转换已完成，直接加载结果")
            return dpo_data
        
        # 继续转换剩余的数据
        remaining_data = sft_data[start_index:]
        
        try:
            if concurrency > 1:
                new_dpo_data = self._convert_concurrent_optimized(
                    remaining_data, start_index, len(sft_data), 
                    checkpoint_path, save_interval, concurrency
                )
            else:
                new_dpo_data = self._convert_sequential_optimized(
                    remaining_data, start_index, len(sft_data),
                    checkpoint_path, save_interval
                )
            
            # 合并数据
            dpo_data.extend(new_dpo_data)
            
            # 保存最终结果
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dpo_data, f, ensure_ascii=False, indent=2)
            
            # 删除检查点文件
            self._delete_checkpoint(checkpoint_path)
            
            if st is not None:
                st.success(f"🎉 转换完成！共转换 {len(dpo_data)} 个样本")
            
            return dpo_data
            
        except Exception as e:
            # 发生错误时保存当前进度
            if dpo_data:
                checkpoint_data = {
                    'completed_count': len(dpo_data),
                    'converted_data': dpo_data,
                    'error': str(e),
                    'timestamp': time.time()
                }
                self._save_checkpoint(checkpoint_path, checkpoint_data)
                
                if st is not None:
                    st.error(f"❌ 转换过程中出现错误: {str(e)}")
                    st.info(f"💾 已保存进度到检查点，下次可以继续转换")
            
            raise e
    
    def _convert_sequential_optimized(
        self, 
        sft_data: List[Dict[str, Any]], 
        start_index: int, 
        total_count: int,
        checkpoint_path: str, 
        save_interval: int
    ) -> List[Dict[str, str]]:
        """
        优化的串行转换
        """
        dpo_data = []
        
        # 创建Streamlit进度条
        progress_bar = None
        status_text = None
        if st is not None:
            progress_bar = st.progress(start_index / total_count)
            status_text = st.empty()
        
        for i, sft_sample in enumerate(tqdm(sft_data, desc="转换SFT到DPO", initial=start_index, total=total_count)):
            try:
                current_index = start_index + i
                
                # 更新进度条
                if progress_bar is not None:
                    progress = (current_index + 1) / total_count
                    progress_bar.progress(progress)
                    status_text.text(f"转换SFT到DPO: {current_index + 1}/{total_count} ({progress:.1%})")
                
                # 转换单个样本
                dpo_sample = self.convert_sft_sample_to_dpo(sft_sample)
                dpo_data.append(dpo_sample)
                
                # 定期保存检查点
                if (i + 1) % save_interval == 0:
                    checkpoint_data = {
                        'completed_count': current_index + 1,
                        'converted_data': dpo_data,
                        'timestamp': time.time()
                    }
                    self._save_checkpoint(checkpoint_path, checkpoint_data)
                    
                    if st is not None:
                        st.info(f"💾 已保存检查点: {current_index + 1}/{total_count}")
                
            except Exception as e:
                print(f"转换样本 {start_index + i} 时出错: {e}")
                continue
        
        return dpo_data
    
    def _convert_concurrent_optimized(
        self, 
        sft_data: List[Dict[str, Any]], 
        start_index: int, 
        total_count: int,
        checkpoint_path: str, 
        save_interval: int, 
        concurrency: int
    ) -> List[Dict[str, str]]:
        """
        优化的并发转换
        """
        dpo_data = [None] * len(sft_data)
        completed_count = 0
        
        # 创建进度条
        progress_bar = None
        status_text = None
        if st is not None:
            progress_bar = st.progress(start_index / total_count)
            status_text = st.empty()
        
        def convert_single_sample(index: int, sft_sample: Dict[str, Any]) -> tuple:
            try:
                dpo_sample = self.convert_sft_sample_to_dpo(sft_sample)
                return index, dpo_sample
            except Exception as e:
                print(f"转换样本 {start_index + index} 时出错: {e}")
                return index, None
        
        # 并发转换
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_index = {
                executor.submit(convert_single_sample, i, sft_sample): i 
                for i, sft_sample in enumerate(sft_data)
            }
            
            for future in as_completed(future_to_index):
                index, dpo_sample = future.result()
                if dpo_sample is not None:
                    dpo_data[index] = dpo_sample
                
                completed_count += 1
                current_total = start_index + completed_count
                
                # 更新进度条
                if progress_bar is not None:
                    progress = current_total / total_count
                    progress_bar.progress(progress)
                    status_text.text(f"并发转换SFT到DPO: {current_total}/{total_count} ({progress:.1%})")
                
                # 定期保存检查点
                if completed_count % save_interval == 0:
                    # 过滤掉None值
                    valid_data = [sample for sample in dpo_data if sample is not None]
                    checkpoint_data = {
                        'completed_count': start_index + len(valid_data),
                        'converted_data': valid_data,
                        'timestamp': time.time()
                    }
                    self._save_checkpoint(checkpoint_path, checkpoint_data)
                    
                    if st is not None:
                        st.info(f"💾 已保存检查点: {current_total}/{total_count}")
        
        # 过滤掉None值，保持顺序
        return [sample for sample in dpo_data if sample is not None]
    
    def convert_folder_sft_to_dpo_optimized(
        self,
        sft_folder_path: str,
        output_folder: str,
        concurrency: int = 1,
        resume_conversion: bool = True,
        save_interval: int = 5
    ) -> Dict[str, Any]:
        """
        优化版本的批量文件夹转换
        """
        # 获取所有JSON文件
        sft_files = []
        for file_name in os.listdir(sft_folder_path):
            if file_name.endswith('.json'):
                sft_files.append(os.path.join(sft_folder_path, file_name))
        
        if not sft_files:
            raise ValueError(f"在文件夹 {sft_folder_path} 中没有找到JSON文件")
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        conversion_results = []
        total_converted = 0
        
        for sft_file in sft_files:
            try:
                file_name = os.path.basename(sft_file)
                output_file = os.path.join(output_folder, f"dpo_{file_name}")
                
                if st is not None:
                    st.info(f"🔄 正在转换文件: {file_name}")
                
                # 使用优化转换方法
                dpo_data = self.convert_sft_dataset_to_dpo_optimized(
                    sft_file, output_file, concurrency, resume_conversion, save_interval
                )
                
                conversion_results.append({
                    'input_file': sft_file,
                    'output_file': output_file,
                    'converted_count': len(dpo_data)
                })
                
                total_converted += len(dpo_data)
                
            except Exception as e:
                print(f"转换文件 {sft_file} 时出错: {e}")
                continue
        
        return {
            'conversion_results': conversion_results,
            'total_converted': total_converted,
            'total_files': len(conversion_results)
        }