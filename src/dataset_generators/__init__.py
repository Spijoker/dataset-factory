# -*- coding: utf-8 -*-
"""
数据集生成器模块
包含SFT（监督微调）和DPO（直接偏好优化）数据集生成功能
"""

from .base_generator import BaseDatasetGenerator
from .sft_generator import SFTDatasetGenerator
from .dpo_generator import DPODatasetGenerator
from .sft_to_dpo_converter import SFTToDPOConverter

__all__ = [
    'BaseDatasetGenerator',
    'SFTDatasetGenerator', 
    'DPODatasetGenerator',
    'SFTToDPOConverter'
]