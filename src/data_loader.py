# -*- coding: utf-8 -*-
"""
数据加载模块，用于加载和处理数据集
"""
import json
import random
from typing import List, Dict, Any, Tuple, Optional
import os
from pathlib import Path
import glob

class DataLoader:
    """
    数据加载器，用于加载和处理数据集
    """
    def __init__(self, input_path: str):
        """
        初始化数据加载器
        
        Args:
            input_path: 数据集文件路径或文件夹路径
        """
        self.input_path = input_path
        self.data = []
        self.file_paths = []
        self.load_data()
    
    def load_data(self) -> None:
        """
        加载数据集（支持单个文件或文件夹）
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"输入路径不存在: {self.input_path}")
        
        # 判断是文件还是文件夹
        if os.path.isfile(self.input_path):
            # 单个文件
            self.file_paths = [self.input_path]
        elif os.path.isdir(self.input_path):
            # 文件夹，查找所有JSON文件
            json_files = glob.glob(os.path.join(self.input_path, "*.json"))
            if not json_files:
                raise FileNotFoundError(f"文件夹中未找到JSON文件: {self.input_path}")
            self.file_paths = json_files
            # 在文件夹中找到JSON文件
        else:
            raise ValueError(f"输入路径既不是文件也不是文件夹: {self.input_path}")
        
        # 加载所有文件的数据
        all_data = []
        for file_path in self.file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        all_data.extend(file_data)
                        # 成功加载数据
                    else:
                        all_data.append(file_data)
                        # 成功加载数据
            except Exception as e:
                # 加载文件失败
                continue
        
        if not all_data:
            raise Exception("未能加载任何有效数据")
        
        self.data = all_data
        # 总共成功加载数据集
    
    def get_random_samples(self, min_samples: int = 3, max_samples: int = 6) -> List[Dict[str, Any]]:
        """
        随机获取指定数量的样本
        
        Args:
            min_samples: 最少样本数量
            max_samples: 最多样本数量
            
        Returns:
            随机样本列表
        """
        if not self.data:
            raise ValueError("数据集为空，无法获取样本")
        
        # 确保样本数量不超过数据集大小
        max_samples = min(max_samples, len(self.data))
        min_samples = min(min_samples, max_samples)
        
        # 随机确定样本数量
        sample_count = random.randint(min_samples, max_samples)
        
        # 随机选择样本
        return random.sample(self.data, sample_count)
    
    def format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """
        格式化示例数据为字符串
        
        Args:
            examples: 示例数据列表
            
        Returns:
            格式化后的示例字符串
        """
        formatted = ""
        for i, example in enumerate(examples):
            formatted += f"示例 {i+1}:\n"
            formatted += f"instruction: {example.get('instruction', '')}\n"
            formatted += f"input: {example.get('input', '')}\n"
            formatted += f"output: {example.get('output', '')}\n\n"
        
        return formatted
    
    def save_data(self, data: List[Dict[str, Any]], output_file: str) -> None:
        """
        保存数据到文件
        
        Args:
            data: 要保存的数据
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 成功保存数据
        except Exception as e:
            raise Exception(f"保存数据失败: {str(e)}")