# -*- coding: utf-8 -*-
"""
配置文件，用于设置数据集路径、模型参数和生成提示词等配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件中的环境变量

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# 确保目录存在（已注释，避免自动生成目录）
# os.makedirs(INPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# 默认输出文件
DEFAULT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "augmented_dataset.json")

# 模型配置
MODEL_TYPE = os.getenv("MODEL_TYPE", "ollama") # 默认模型类型
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1:8b")  # 默认使用的Ollama模型
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 生成配置
SAMPLE_MIN = 3  # 最少示例数量
SAMPLE_MAX = 6  # 最多示例数量
GENERATION_NUM = 50  # 默认生成的数据条目数量

# 提示词模板
INSTRUCTION_PROMPT = """
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

INPUT_PROMPT = """
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

OUTPUT_PROMPT = """
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