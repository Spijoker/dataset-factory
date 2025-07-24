# 🤖 智能数据集生成器 (AI Dataset Generator)

这是一个基于大语言模型的智能数据集生成工具，可以通过已有的数据集生成更多高质量的合成数据，用于模型微调和训练。

## ✨ 功能特点

- 🎯 **多类型数据集**：支持SFT、DPO和SFT转DPO三种数据集类型
  - **SFT模式**：生成监督微调数据集（instruction+input+output）
  - **DPO模式**：生成直接偏好优化数据集（instruction+input+chosen+rejected）
  - **SFT转DPO**：将现有SFT数据集转换为DPO格式
- 🧠 **多模型支持**：兼容Ollama和OpenAI兼容的API
- 📁 **灵活输入**：支持单个文件或文件夹输入（自动处理文件夹中所有JSON文件）
- 📤 **自定义输出**：可指定输出文件夹和文件名
- 🖥️ **Web界面**：现代化的Streamlit界面，功能全面
- 📊 **实时监控**：生成过程实时显示进度和状态
- 📈 **数据分析**：自动分析生成数据的统计信息
- ✂️ **数据分割**：自动分割训练集和验证集
- 💾 **便捷导出**：支持JSON格式的数据导出

## 项目结构

```
dataset-factory/
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── app.py                    # Web界面主程序
├── run_app.py                # Web界面启动脚本
├── config.py                 # 配置文件
├── prompt_config.py          # 提示词配置文件
├── .streamlit/               # Streamlit配置
│   └── config.toml           # Streamlit配置文件
└── src/                      # 源代码
    ├── __init__.py
    ├── data_loader.py        # 数据加载模块
    ├── data_generator.py     # 数据生成模块
    ├── model_caller.py       # 模型调用模块
    ├── utils.py              # 工具函数
    ├── prompt_config.py      # 提示词配置管理
    └── dataset_generators/   # 模块化生成器
        ├── __init__.py
        ├── base_generator.py # 基础生成器
        ├── sft_generator.py  # SFT生成器
        ├── dpo_generator.py  # DPO生成器
        └── sft_to_dpo_converter.py # SFT转DPO转换器
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动Web界面

**方式一：使用启动脚本（推荐）**
```bash
python run_app.py
```

**方式二：直接启动Streamlit**
```bash
streamlit run app.py
```

启动后，浏览器会自动打开 http://localhost:8501，如果没有自动打开，请手动访问该地址。

## 🌐 Web界面功能

### 主要功能区域

1. **📁 数据集管理**
   - 支持文件和文件夹输入
   - 数据集预览和统计
   - 自动处理JSON格式数据

2. **🧠 模型配置**
   - Ollama模型支持
   - OpenAI兼容API支持
   - 实时模型连接测试

3. **🎯 生成模式选择**
   - **SFT模式**：生成监督微调数据集
   - **DPO模式**：生成直接偏好优化数据集
   - **SFT转DPO**：转换现有SFT数据集

4. **📊 实时监控**
   - 生成进度实时显示
   - 错误处理和重试机制
   - 生成质量统计

5. **📈 结果分析**
   - 数据统计信息
   - 样本预览和查看
   - JSON格式导出

### Web界面特色

- 🎨 **现代化设计**：采用渐变色彩和卡片式布局
- 📱 **响应式界面**：适配不同屏幕尺寸
- ⚡ **实时反馈**：操作结果即时显示
- 🔄 **进度可视化**：生成过程清晰可见
- 💾 **一键导出**：支持JSON格式下载

## ⚙️ 配置说明

### 环境变量配置

在使用前，请根据需要配置以下环境变量（可创建 `.env` 文件）：

```bash
# 模型配置
MODEL_TYPE=ollama                    # 模型类型：ollama 或 openai_compatible
MODEL_NAME=deepseek-r1:8b            # 默认模型名称

# OpenAI兼容API配置（如果使用）
OPENAI_API_KEY=your_api_key_here     # API密钥
OPENAI_BASE_URL=your_base_url_here   # API基础URL
```

### 配置文件 (config.py)

主要配置项：
- `MODEL_TYPE`：默认模型类型
- `MODEL_NAME`：默认模型名称
- `GENERATION_NUM`：默认生成数量
- `SAMPLE_MIN/MAX`：示例数量范围
- 各种提示词模板配置

## 🔧 高级功能

### 数据集分析

生成完成后自动分析：
- 样本数量统计
- 文本长度分布
- 内容质量评估
- 重复度检测

### 数据集分割

自动分割为训练集和验证集：
- 可自定义分割比例
- 保持数据分布均衡
- 生成独立的文件

### 批量处理

支持批量生成：
- 多个数据集并行处理
- 断点续传功能
- 错误恢复机制

## 🛠️ 扩展开发

### 添加新模型支持

1. 在 `src/model_caller.py` 中添加新的模型类
2. 实现 `ModelCaller` 接口
3. 在 `ModelCallerFactory` 中注册新模型

### 自定义提示词

1. 修改 `config.py` 中的提示词模板
2. 支持变量替换和条件逻辑
3. 可针对不同任务优化

### 界面定制

1. 修改 `app.py` 中的CSS样式
2. 添加新的功能组件
3. 自定义主题和布局

## 📝 使用说明

### 基本使用流程

1. **准备数据集**：准备JSON格式的种子数据集，包含 `instruction`、`input`、`output` 字段
2. **启动界面**：运行 `python run_app.py` 启动Web界面
3. **选择输入**：在界面中选择数据集文件或文件夹
4. **配置模型**：选择模型类型（Ollama或OpenAI兼容）并配置相关参数
5. **选择模式**：选择生成模式（SFT、DPO或SFT转DPO）
6. **设置参数**：配置生成数量、示例范围等参数
7. **开始生成**：点击生成按钮，实时查看进度
8. **查看结果**：生成完成后查看统计信息和样本预览
9. **导出数据**：下载生成的数据集文件

### 数据格式要求

输入数据集应为JSON格式，每条数据包含以下字段：

```json
[
  {
    "instruction": "请解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
  },
  {
    "instruction": "翻译以下英文",
    "input": "Hello, how are you?",
    "output": "你好，你好吗？"
  }
]
```

## ❓ 常见问题

### Q: Web界面无法访问怎么办？

A:
1. 检查是否已安装所有依赖：`pip install -r requirements.txt`
2. 确保端口8501没有被占用
3. 尝试手动访问 http://localhost:8501
4. 检查防火墙设置，确保允许本地连接
5. 查看控制台错误信息

### Q: 模型连接失败怎么办？

A:
1. **Ollama模型**：确保Ollama服务正在运行，检查模型名称是否正确
2. **OpenAI兼容API**：检查API密钥和基础URL是否正确配置
3. 在Web界面中测试模型连接
4. 查看错误提示信息

### Q: 如何提高生成质量？

A: 
1. 使用高质量的种子数据集
2. 调整示例数量（sample_min/max）
3. 优化提示词模板
4. 选择更强的模型

### Q: 生成速度太慢怎么办？

A:
1. 使用本地部署的模型（如Ollama）
2. 减少示例数量
3. 调整生成延迟参数
4. 使用更快的模型

### Q: 如何处理生成错误？

A:
1. 检查模型连接状态
2. 验证输入数据格式
3. 查看错误日志
4. 使用断点续传功能

### Q: 支持哪些数据格式？

A:
- 输入：JSON格式（Alpaca格式）
- 输出：JSON格式
- 字段：instruction, input, output

### Q: 如何自定义提示词？

A:
1. 编辑 `config.py` 中的提示词模板
2. 使用变量占位符（如 {examples}）
3. 根据任务特点调整提示词
4. 测试不同提示词的效果

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**享受智能数据集生成的乐趣！** 🚀