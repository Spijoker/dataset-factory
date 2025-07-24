# -*- coding: utf-8 -*-
"""
工具函数模块，提供一些通用的辅助功能
"""
import os
import json
import random
from typing import List, Dict, Any, Optional
import time
from datetime import datetime


def setup_directories(dirs: List[str]) -> None:
    """
    确保目录存在
    
    Args:
        dirs: 目录列表
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        # 确保目录存在


def get_timestamp() -> str:
    """
    获取当前时间戳字符串
    
    Returns:
        时间戳字符串，格式：YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def merge_datasets(file_paths: List[str], output_file: str) -> None:
    """
    合并多个数据集文件
    
    Args:
        file_paths: 数据集文件路径列表
        output_file: 输出文件路径
    """
    merged_data = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            # 文件不存在，跳过
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                    # 成功加载数据
                else:
                    # 文件不是有效的JSON数组，跳过
                    pass
        except Exception as e:
            # 加载文件时出错
            pass
    
    # 保存合并后的数据
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        # 成功保存合并数据
    except Exception as e:
        # 保存合并数据时出错
        pass


def split_dataset(file_path: str, train_ratio: float = 0.8, 
                 output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    将数据集分割为训练集和验证集
    
    Args:
        file_path: 数据集文件路径
        train_ratio: 训练集比例
        output_dir: 输出目录，默认为文件所在目录
        
    Returns:
        包含训练集和验证集文件路径的字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise Exception(f"加载数据集失败: {str(e)}")
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算分割点
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 构建输出文件路径
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = get_timestamp()
    train_file = os.path.join(output_dir, f"{base_name}_train_{timestamp}.json")
    val_file = os.path.join(output_dir, f"{base_name}_val_{timestamp}.json")
    
    # 保存训练集和验证集
    try:
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        # 成功保存训练集
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        # 成功保存验证集
    except Exception as e:
        raise Exception(f"保存数据集失败: {str(e)}")
    
    return {
        "train": train_file,
        "val": val_file
    }


def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """
    分析数据集的基本统计信息
    
    Args:
        file_path: 数据集文件路径
        
    Returns:
        包含统计信息的字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    # 加载数据集
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise Exception(f"加载数据集失败: {str(e)}")
    
    # 基本统计信息
    stats = {
        "total_samples": len(data),
        "instruction_length": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        },
        "input_length": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        },
        "output_length": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        }
    }
    
    # 计算长度统计
    total_instruction_len = 0
    total_input_len = 0
    total_output_len = 0
    
    for item in data:
        # 指令长度
        instr_len = len(item.get("instruction", ""))
        stats["instruction_length"]["min"] = min(stats["instruction_length"]["min"], instr_len)
        stats["instruction_length"]["max"] = max(stats["instruction_length"]["max"], instr_len)
        total_instruction_len += instr_len
        
        # 输入长度
        input_len = len(item.get("input", ""))
        stats["input_length"]["min"] = min(stats["input_length"]["min"], input_len)
        stats["input_length"]["max"] = max(stats["input_length"]["max"], input_len)
        total_input_len += input_len
        
        # 输出长度
        output_len = len(item.get("output", ""))
        stats["output_length"]["min"] = min(stats["output_length"]["min"], output_len)
        stats["output_length"]["max"] = max(stats["output_length"]["max"], output_len)
        total_output_len += output_len
    
    # 计算平均长度
    if len(data) > 0:
        stats["instruction_length"]["avg"] = total_instruction_len / len(data)
        stats["input_length"]["avg"] = total_input_len / len(data)
        stats["output_length"]["avg"] = total_output_len / len(data)
    
    return stats