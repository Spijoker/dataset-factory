# -*- coding: utf-8 -*-
"""
DPO（直接偏好优化）数据集生成器
支持完整模式和固定指令模式
"""

import json
from typing import List, Dict, Any, Optional

from .base_generator import BaseDatasetGenerator
from ..model_caller import extract_content_between_backticks


class DPODatasetGenerator(BaseDatasetGenerator):
    """
    DPO（直接偏好优化）数据集生成器
    
    支持两种模式：
    1. 完整模式：生成instruction、input、chosen、rejected
    2. 固定指令模式：使用固定instruction，生成input、chosen、rejected
    
    DPO数据集格式：
    [
        {
            "instruction": "人类指令（必填）",
            "input": "人类输入（选填）",
            "chosen": "优质回答（必填）",
            "rejected": "劣质回答（必填）"
        }
    ]
    """
    
    def __init__(
        self,
        model_caller,
        data_loader,
        instruction_prompt: str,
        input_prompt: str,
        chosen_prompt: str,
        rejected_prompt: str,
        sample_min: int = 3,
        sample_max: int = 6
    ):
        """
        初始化DPO数据集生成器
        
        Args:
            model_caller: 模型调用器
            data_loader: 数据加载器
            instruction_prompt: 生成instruction的提示模板
            input_prompt: 生成input的提示模板
            chosen_prompt: 生成chosen（优质回答）的提示模板
            rejected_prompt: 生成rejected（劣质回答）的提示模板
            sample_min: 最少示例数量
            sample_max: 最多示例数量
        """
        super().__init__(model_caller, data_loader, sample_min, sample_max)
        self.instruction_prompt = instruction_prompt
        self.input_prompt = input_prompt
        self.chosen_prompt = chosen_prompt
        self.rejected_prompt = rejected_prompt
    
    def get_dataset_format_description(self) -> str:
        """
        获取数据集格式描述
        """
        return "DPO（直接偏好优化）"
    
    def generate_sample(self, mode: str = "complete", fixed_instruction: Optional[str] = None) -> Dict[str, str]:
        """
        生成单个DPO样本
        
        Args:
            mode: 生成模式，"complete"为完整模式，"input_output"为固定指令模式
            fixed_instruction: 固定指令（仅在input_output模式下使用）
            
        Returns:
            包含instruction, input, chosen, rejected的字典
        """
        if mode == "complete":
            return self.generate_complete_sample()
        elif mode == "input_output":
            return self.generate_input_output_sample(fixed_instruction)
        else:
            raise ValueError(f"不支持的生成模式: {mode}")
    
    def generate_instructions(self, num_to_generate: int = 1) -> List[str]:
        """
        生成新的instructions
        
        Args:
            num_to_generate: 要生成的instruction数量
            
        Returns:
            生成的instruction列表
        """
        # 获取随机示例
        examples = self.get_random_examples()
        formatted_examples = self.format_examples(examples)
        
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
            # 再次尝试直接处理原始响应
            raw_lines = [line.strip() for line in response.split('\n') if line.strip()]
            for line in raw_lines:
                # 过滤掉可能的非 instruction 行
                if not line.lower().startswith('示例') and \
                   not line.lower().startswith('以下是') and \
                   not line.lower().startswith('这是') and \
                   not line.lower().startswith('json') and \
                   line not in instructions:  # 避免重复添加
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
        examples = self.get_random_examples()
        formatted_examples = self.format_examples(examples)
        
        # 构建提示词
        prompt = self.input_prompt.format(
            instruction=instruction,
            examples=formatted_examples
        )
        
        # 调用模型生成
        response = self.model_caller.generate(prompt)
        
        # 提取生成的input
        return extract_content_between_backticks(response)
    
    def generate_chosen(self, instruction: str, input_text: str) -> str:
        """
        为给定的instruction和input生成chosen（优质回答）
        
        Args:
            instruction: 指令
            input_text: 输入
            
        Returns:
            生成的chosen（优质回答）
        """
        # 获取随机示例
        examples = self.get_random_examples()
        formatted_examples = self.format_examples(examples)
        
        # 构建提示词
        prompt = self.chosen_prompt.format(
            instruction=instruction,
            input=input_text,
            examples=formatted_examples
        )
        
        # 调用模型生成
        response = self.model_caller.generate(prompt)
        
        # 提取生成的chosen
        return extract_content_between_backticks(response)
    
    def generate_rejected(self, instruction: str, input_text: str, chosen: str) -> str:
        """
        为给定的instruction、input和chosen生成rejected（劣质回答）
        
        Args:
            instruction: 指令
            input_text: 输入
            chosen: 优质回答
            
        Returns:
            生成的rejected（劣质回答）
        """
        # 获取随机示例
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
    
    def generate_input_output_sample(self, instruction: Optional[str] = None) -> Dict[str, str]:
        """
        为给定的instruction生成input、chosen和rejected（固定指令模式）
        
        Args:
            instruction: 固定的指令，如果为None或空字符串则从原始数据集中随机选择
            
        Returns:
            包含instruction, input, chosen, rejected的字典
        """
        # 如果没有提供instruction或instruction为空字符串，从原始数据集中随机选择一个
        if not instruction or not instruction.strip():
            sample = self.get_random_examples()[0]
            instruction = sample.get('instruction', '')
            if not instruction:
                raise ValueError("原始数据集中没有找到有效的instruction")
        
        # 生成input
        input_text = self.generate_input(instruction)
        
        # 生成chosen（优质回答）
        chosen = self.generate_chosen(instruction, input_text)
        
        # 生成rejected（劣质回答）
        rejected = self.generate_rejected(instruction, input_text, chosen)
        
        return {
            "instruction": instruction,
            "input": input_text,
            "chosen": chosen,
            "rejected": rejected
        }
    
    def generate_complete_sample(self) -> Dict[str, str]:
        """
        生成完整的DPO样本（instruction, input, chosen, rejected）
        
        Returns:
            包含instruction, input, chosen, rejected的字典
        """
        # 生成instruction
        instructions = self.generate_instructions(1)
        if not instructions:
            raise ValueError("生成instruction失败")
        instruction = instructions[0]
        
        # 生成input
        input_text = self.generate_input(instruction)
        
        # 生成chosen（优质回答）
        chosen = self.generate_chosen(instruction, input_text)
        
        # 生成rejected（劣质回答）
        rejected = self.generate_rejected(instruction, input_text, chosen)
        
        return {
            "instruction": instruction,
            "input": input_text,
            "chosen": chosen,
            "rejected": rejected
        }