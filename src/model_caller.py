# -*- coding: utf-8 -*-
"""
模型调用模块，用于调用大语言模型
"""
from typing import Dict, List, Any, Optional, Union
import re

class ModelCaller:
    """
    模型调用器基类，定义通用接口
    """
    def __init__(self, model_name: str):
        """
        初始化模型调用器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示词
            
        Returns:
            生成的文本
        """
        raise NotImplementedError("子类必须实现此方法")


class OllamaModelCaller(ModelCaller):
    """
    Ollama模型调用器
    """
    def __init__(self, model_name: str):
        """
        初始化Ollama模型调用器
        
        Args:
            model_name: Ollama模型名称
        """
        super().__init__(model_name)
        try:
            from ollama import chat
            self.chat = chat
            # 成功初始化Ollama模型
        except ImportError:
            raise ImportError("请安装ollama包: pip install ollama")
    
    def generate(self, prompt: str) -> str:
        """
        使用Ollama模型生成文本
        
        Args:
            prompt: 提示词
            
        Returns:
            生成的文本
        """
        try:
            response = self.chat(
                model=self.model_name, 
                messages=[
                    {
                        'role': 'user',
                        'content': f"{prompt},'/no_think'",
                    }
                ],
                options={
                    'stream': False,
                    'think': False
                }
            )
            return response['message']['content']
        except Exception as e:
            # Ollama模型调用失败
            return ""


class OpenAICompatibleModelCaller(ModelCaller):
    """
    OpenAI 兼容的 API 模型调用器
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 OpenAI 兼容的 API 模型调用器
        
        Args:
            model_name: 模型名称
            api_key: API Key
            base_url: API Base URL
        """
        super().__init__(model_name)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            # 成功初始化 OpenAI 兼容模型
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
        except Exception as e:
            raise RuntimeError(f"初始化 OpenAI 兼容模型失败: {str(e)}")

    def generate(self, prompt: str) -> str:
        """
        使用 OpenAI 兼容模型生成文本
        
        Args:
            prompt: 提示词
            
        Returns:
            生成的文本
        """
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7, # 可以根据需要调整
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            # OpenAI 兼容模型调用失败
            return ""


class ModelCallerFactory:
    """
    模型调用器工厂，用于创建不同类型的模型调用器
    """
    @staticmethod
    def create(model_type: str, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> ModelCaller:
        """
        创建模型调用器
        
        Args:
            model_type: 模型类型，如 'ollama', 'openai_compatible'
            model_name: 模型名称
            api_key: (可选) API Key，用于 OpenAI 兼容模型
            base_url: (可选) API Base URL，用于 OpenAI 兼容模型
            
        Returns:
            模型调用器实例
        """
        if model_type.lower() == 'ollama':
            model_caller = OllamaModelCaller(model_name)
        elif model_type.lower() == 'openai_compatible':
            # 如果没有提供api_key和base_url，用户需要手动配置
            model_caller = OpenAICompatibleModelCaller(model_name, api_key, base_url)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 测试模型连通性
        ModelCallerFactory.test_model_connectivity(model_caller)
        return model_caller
    
    @staticmethod
    def test_model_connectivity(model_caller: ModelCaller) -> bool:
        """
        测试模型连通性
        
        Args:
            model_caller: 模型调用器实例
            
        Returns:
            是否连通成功
            
        Raises:
            RuntimeError: 当模型连通性测试失败时
        """
        try:
            # 正在测试模型连通性
            test_response = model_caller.generate("测试连接，请回复'连接成功'")
            
            if not test_response or test_response.strip() == "":
                raise RuntimeError(f"模型 {model_caller.model_name} 连通性测试失败：模型无响应")
            
            # 模型连通性测试成功
            return True
            
        except Exception as e:
            error_msg = f"模型 {model_caller.model_name} 连通性测试失败：{str(e)}"
            if isinstance(model_caller, OllamaModelCaller):
                error_msg += "\n请检查：\n1. Ollama服务是否正在运行\n2. 模型名称是否正确\n3. 模型是否已下载"
            elif isinstance(model_caller, OpenAICompatibleModelCaller):
                error_msg += "\n请检查：\n1. API Key是否正确\n2. Base URL是否正确\n3. 模型名称是否正确\n4. 网络连接是否正常"
            
            raise RuntimeError(error_msg)


def extract_content_between_backticks(text: str) -> str:
    """
    提取三个反引号之间的内容，并尝试处理 JSON 格式。
    
    Args:
        text: 包含反引号的文本
        
    Returns:
        提取的内容
    """
    import json
    # 优先匹配 ```json\n{...}\n``` 格式
    json_pattern = r'```json\s*(\{.*\})\s*```'
    json_match = re.search(json_pattern, text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # 其次匹配 ```\n{...}\n``` 格式
    general_pattern = r'```\s*(.*?)\s*```'
    general_match = re.search(general_pattern, text, re.DOTALL)
    if general_match:
        content = general_match.group(1).strip()
        # 如果提取到的内容以 "json\n" 开头，尝试去除
        if content.lower().startswith("json\n"):
            content = content[5:].strip()
        return content

    # 如果没有匹配到反引号，尝试直接处理文本
    # 检查是否以 "json\n" 开头，并尝试解析为 JSON
    if text.lower().startswith("json\n"):
        potential_json_str = text[5:].strip()
        try:
            # 尝试解析为 JSON，如果成功，返回 JSON 字符串
            json.loads(potential_json_str)
            return potential_json_str
        except json.JSONDecodeError:
            pass # 不是有效的 JSON，继续处理

    # 尝试去除常见的前缀和后缀
    cleaned_text = text
    prefixes = ["以下是生成的", "这是", "生成的", "以下是", "json"]
    for prefix in prefixes:
        if cleaned_text.lower().startswith(prefix.lower()):
            cleaned_text = cleaned_text[len(prefix):].lstrip()
    
    # 去除可能的后缀
    suffixes = ["希望这对你有帮助", "希望这能满足你的需求", "如有需要"]
    for suffix in suffixes:
        if cleaned_text.lower().endswith(suffix.lower()):
            cleaned_text = cleaned_text[:cleaned_text.lower().find(suffix.lower())].rstrip()
    
    return cleaned_text.strip()