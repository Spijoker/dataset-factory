# -*- coding: utf-8 -*-
"""
提示词配置管理模块
用于管理用户自定义的提示词模板
"""
import json
import os
from typing import Dict, List, Optional
from pathlib import Path

# 提示词配置文件路径
PROMPT_CONFIG_FILE = os.path.join(Path(__file__).resolve().parent, "prompt_configs.json")

# 默认提示词模板
DEFAULT_PROMPTS = {
    "instruction": {
        "name": "默认指令生成提示词",
        "description": "用于生成新的instruction的默认提示词",
        "template": """
你是一个专业的数据集生成助手。我需要你帮我生成一些新的指令(instruction)，用于扩充我的数据集。

我会给你一些现有数据集中的示例，每个示例包含instruction、input和output三个字段。
请参考这些示例，生成{num_to_generate}个新的、多样化的instruction。

生成的instruction应该：
1. 与示例中的instruction风格一致
2. 内容符合常识且合理
3. 与示例中的instruction内容所引导的目标是一致的，在此基础上进行多种变化
4. 内容真实、合理，避免虚构或不合逻辑的内容，要让模型能够理解

以下是示例数据：
{examples}

请直接生成{num_to_generate}个新的instruction，每个instruction用三个反引号(```)包裹，不要有编号，不要有额外的解释，不要包含任何格式说明（如'json'）。
"""
    },
    "input": {
        "name": "默认输入生成提示词",
        "description": "用于生成input的默认提示词",
        "template": """
你是一个专业的数据集生成助手。我需要你帮我为给定的instruction生成合适的input，用于扩充我的数据集。

我会给你一些现有数据集中的示例，每个示例包含instruction、input和output三个字段。
请参考这些示例，为以下instruction生成一个合适的input：

指令(instruction): {instruction}

生成的input应该：
1. 与示例中的input风格一致
2. 符合instruction的要求和上下文
3. 提供足够的信息让模型能够基于instruction生成合理的output
4. 内容真实、合理，避免虚构或不合逻辑的内容

以下是示例数据：
{examples}

请直接生成input，用三个反引号(```)包裹，不要有额外的解释，不要包含任何格式说明（如'json'）。
"""
    },
    "output": {
        "name": "默认输出生成提示词",
        "description": "用于生成output的默认提示词",
        "template": """
你是一个专业的数据集生成助手。我需要你帮我为给定的instruction和input生成合适的output，用于扩充我的数据集。

我会给你一些现有数据集中的示例，每个示例包含instruction、input和output三个字段。
请参考这些示例，为以下instruction和input生成一个合适的output：

指令(instruction): {instruction}
输入(input): {input}

生成的output应该：
1. 与示例中的output风格一致
2. 严格按照instruction的要求来响应input
3. 内容完整、准确、有逻辑性
4. 如果instruction要求特定的输出格式，请严格遵循

以下是示例数据：
{examples}

请直接生成output，用三个反引号(```)包裹，不要有额外的解释，不要包含任何格式说明（如'json'）。
"""
    },
    "chosen": {
        "name": "默认优质回答生成提示词",
        "description": "用于生成DPO数据集中chosen字段的默认提示词",
        "template": """
你是一个专业的数据集生成助手。我需要你帮我为给定的instruction和input生成一个高质量的回答(chosen)，用于DPO数据集。

我会给你一些现有数据集中的示例，请参考这些示例，为以下instruction和input生成一个优质的回答：

指令(instruction): {instruction}
输入(input): {input}

生成的chosen回答应该：
1. 与示例中的回答风格一致
2. 严格按照instruction的要求来响应input
3. 内容完整、准确、有逻辑性、有帮助性
4. 语言流畅，表达清晰
5. 如果instruction要求特定的输出格式，请严格遵循

以下是示例数据：
{examples}

请直接生成chosen回答，用三个反引号(```)包裹，不要有额外的解释，不要包含任何格式说明（如'json'）。
"""
    },
    "rejected": {
        "name": "默认劣质回答生成提示词",
        "description": "用于生成DPO数据集中rejected字段的默认提示词",
        "template": """
你是一个专业的数据集生成助手。我需要你帮我为给定的instruction和input生成一个相对较差的回答(rejected)，用于DPO数据集。

我会给你一些现有数据集中的示例，请参考这些示例，为以下instruction和input生成一个质量较差的回答：

指令(instruction): {instruction}
输入(input): {input}

生成的rejected回答应该：
1. 在形式上看起来合理，但质量明显低于优质回答
2. 可能包含以下问题之一：
   - 回答不够完整或详细
   - 逻辑不够清晰
   - 没有完全理解instruction的要求
   - 信息不够准确或有轻微错误
   - 表达不够流畅
3. 避免生成完全错误或有害的内容
4. 确保回答仍然是可理解的

以下是示例数据：
{examples}

请直接生成rejected回答，用三个反引号(```)包裹，不要有额外的解释，不要包含任何格式说明（如'json'）。
"""
    }
}

class PromptConfigManager:
    """提示词配置管理器"""
    
    def __init__(self):
        self.config_file = PROMPT_CONFIG_FILE
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict:
        """加载提示词配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_prompts = json.load(f)
                # 使用保存的提示词，如果没有则使用默认值
                prompts = {
                    'sft': saved_prompts.get('sft', {
                        'instruction': DEFAULT_PROMPTS['instruction']['template'],
                        'input': DEFAULT_PROMPTS['input']['template'],
                        'output': DEFAULT_PROMPTS['output']['template']
                    }),
                    'dpo': saved_prompts.get('dpo', {
                        'instruction': DEFAULT_PROMPTS['instruction']['template'],
                        'input': DEFAULT_PROMPTS['input']['template'],
                        'chosen': DEFAULT_PROMPTS['chosen']['template'],
                        'rejected': DEFAULT_PROMPTS['rejected']['template']
                    }),
                    'sft_to_dpo': saved_prompts.get('sft_to_dpo', {
                        'rejected': DEFAULT_PROMPTS['rejected']['template']
                    })
                }
                return prompts
            except Exception as e:
                # 加载提示词配置失败
                return self._get_default_prompts()
        else:
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict:
        """获取默认提示词配置"""
        return {
            'sft': {
                'instruction': DEFAULT_PROMPTS['instruction']['template'],
                'input': DEFAULT_PROMPTS['input']['template'],
                'output': DEFAULT_PROMPTS['output']['template']
            },
            'dpo': {
                'instruction': DEFAULT_PROMPTS['instruction']['template'],
                'input': DEFAULT_PROMPTS['input']['template'],
                'chosen': DEFAULT_PROMPTS['chosen']['template'],
                'rejected': DEFAULT_PROMPTS['rejected']['template']
            },
            'sft_to_dpo': {
                'rejected': DEFAULT_PROMPTS['rejected']['template']
            }
        }
    
    def _save_prompts(self):
        """保存提示词配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.prompts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 保存提示词配置失败
            pass
    
    def get_prompt_types(self) -> List[str]:
        """获取所有提示词类型"""
        return list(self.prompts.keys())
    
    def get_prompts_by_type(self, prompt_type: str) -> Dict:
        """获取指定类型的所有提示词"""
        return self.prompts.get(prompt_type, {})
    
    def get_prompt_names_by_type(self, prompt_type: str) -> List[str]:
        """获取指定类型的所有提示词名称"""
        prompts = self.get_prompts_by_type(prompt_type)
        return [config.get('name', config_id) for config_id, config in prompts.items()]
    
    def get_prompt_template(self, prompt_type: str, prompt_id: str = "default") -> Optional[str]:
        """获取指定提示词模板"""
        prompts = self.get_prompts_by_type(prompt_type)
        if prompt_id in prompts:
            return prompts[prompt_id].get('template', '')
        # 如果找不到指定ID，返回默认提示词
        if 'default' in prompts:
            return prompts['default'].get('template', '')
        return None
    
    def add_prompt(self, prompt_type: str, prompt_id: str, name: str, description: str, template: str) -> bool:
        """添加新的提示词"""
        try:
            if prompt_type not in self.prompts:
                self.prompts[prompt_type] = {}
            
            self.prompts[prompt_type][prompt_id] = {
                'name': name,
                'description': description,
                'template': template
            }
            
            self._save_prompts()
            return True
        except Exception as e:
            # 添加提示词失败
            return False
    
    def update_prompt(self, prompt_type: str, prompt_id: str, name: str, description: str, template: str) -> bool:
        """更新提示词"""
        try:
            if prompt_type in self.prompts and prompt_id in self.prompts[prompt_type]:
                self.prompts[prompt_type][prompt_id] = {
                    'name': name,
                    'description': description,
                    'template': template
                }
                self._save_prompts()
                return True
            return False
        except Exception as e:
            # 更新提示词失败
            return False
    
    def delete_prompt(self, prompt_type: str, prompt_id: str) -> bool:
        """删除提示词（不能删除默认提示词）"""
        try:
            if prompt_type in self.prompts and prompt_id in self.prompts[prompt_type] and prompt_id != "default":
                del self.prompts[prompt_type][prompt_id]
                self._save_prompts()
                return True
            return False
        except Exception as e:
            # 删除提示词失败
            return False
    
    def get_prompt_info(self, prompt_type: str, prompt_id: str) -> Optional[Dict]:
        """获取提示词详细信息"""
        prompts = self.get_prompts_by_type(prompt_type)
        return prompts.get(prompt_id)
    
    # SFT提示词管理方法
    def get_sft_prompts(self) -> Dict[str, str]:
        """获取SFT提示词配置"""
        return self.prompts.get('sft', self._get_default_prompts()['sft'])
    
    def update_sft_prompts(self, instruction_prompt: str, input_prompt: str, output_prompt: str):
        """更新SFT提示词配置"""
        self.prompts['sft'] = {
            'instruction': instruction_prompt,
            'input': input_prompt,
            'output': output_prompt
        }
        self._save_prompts()
    
    def reset_sft_prompts(self):
        """重置SFT提示词为默认值"""
        self.prompts['sft'] = self._get_default_prompts()['sft']
        self._save_prompts()
    
    # DPO提示词管理方法
    def get_dpo_prompts(self) -> Dict[str, str]:
        """获取DPO提示词配置"""
        return self.prompts.get('dpo', self._get_default_prompts()['dpo'])
    
    def update_dpo_prompts(self, instruction_prompt: str, input_prompt: str, chosen_prompt: str, rejected_prompt: str):
        """更新DPO提示词配置"""
        self.prompts['dpo'] = {
            'instruction': instruction_prompt,
            'input': input_prompt,
            'chosen': chosen_prompt,
            'rejected': rejected_prompt
        }
        self._save_prompts()
    
    def reset_dpo_prompts(self):
        """重置DPO提示词为默认值"""
        self.prompts['dpo'] = self._get_default_prompts()['dpo']
        self._save_prompts()
    
    # SFT转DPO提示词管理方法
    def get_sft_to_dpo_prompts(self) -> Dict[str, str]:
        """获取SFT转DPO提示词配置"""
        return self.prompts.get('sft_to_dpo', self._get_default_prompts()['sft_to_dpo'])
    
    def update_sft_to_dpo_prompts(self, rejected_prompt: str):
        """更新SFT转DPO提示词配置"""
        self.prompts['sft_to_dpo'] = {
            'rejected': rejected_prompt
        }
        self._save_prompts()
    
    def reset_sft_to_dpo_prompts(self):
        """重置SFT转DPO提示词为默认值"""
        self.prompts['sft_to_dpo'] = self._get_default_prompts()['sft_to_dpo']
        self._save_prompts()

# 全局提示词管理器实例
prompt_manager = PromptConfigManager()