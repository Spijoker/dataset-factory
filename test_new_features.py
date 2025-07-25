#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新功能的脚本
包括提示词版本管理、内存优化和断点续传功能
"""

import os
import json
import tempfile
from pathlib import Path

# 测试提示词版本管理
def test_prompt_version_management():
    print("=== 测试提示词版本管理 ===")
    
    from config.prompt_config import prompt_manager
    
    # 测试获取提示词
    sft_prompts = prompt_manager.get_sft_prompts()
    print(f"SFT提示词: {list(sft_prompts.keys())}")
    
    # 测试更新提示词并保存版本
    original_instruction = sft_prompts['instruction']
    new_instruction = "这是一个测试的新指令模板"
    
    print(f"原始指令: {original_instruction[:50]}...")
    
    # 更新提示词（会自动保存版本）
    prompt_manager.update_prompt('sft', 'instruction', new_instruction, save_version=True)
    print(f"更新后指令: {new_instruction}")
    
    # 查看版本历史
    versions = prompt_manager.get_prompt_versions('sft', 'instruction')
    print(f"版本历史数量: {len(versions)}")
    
    if versions:
        latest_version = versions[0]
        print(f"最新版本: {latest_version['timestamp']} - {latest_version['content'][:30]}...")
    
    # 恢复原始版本
    if versions:
        prompt_manager.restore_prompt_version('sft', 'instruction', versions[0]['version_id'])
        print("已恢复到原始版本")
    
    print("提示词版本管理测试完成\n")

# 测试内存优化和断点续传
def test_optimized_generator():
    print("=== 测试内存优化和断点续传 ===")
    
    try:
        from src.optimized_data_generator import OptimizedDataGenerator
        from src.data_loader import DataLoader
        from src.model_caller import ModelCallerFactory
        from src.dataset_generators.sft_generator import SFTDatasetGenerator
        from config.prompt_config import prompt_manager
        
        # 创建测试数据
        test_data = [
            {"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是人工智能的一个分支..."},
            {"instruction": "描述深度学习的概念", "input": "", "output": "深度学习是机器学习的一个子集..."}
        ]
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            test_file = f.name
        
        print(f"创建测试文件: {test_file}")
        
        # 创建临时输出目录
        output_dir = tempfile.mkdtemp()
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        output_file = os.path.join(output_dir, "test_output.json")
        
        print(f"输出目录: {output_dir}")
        print(f"检查点目录: {checkpoint_dir}")
        
        # 初始化组件
        data_loader = DataLoader(test_file)
        
        # 使用模拟的模型调用器（避免实际API调用）
        class MockModelCaller:
            def call_model(self, prompt, **kwargs):
                return "这是一个模拟的模型响应"
        
        model_caller = MockModelCaller()
        
        # 获取提示词
        prompts = prompt_manager.get_sft_prompts()
        
        # 创建SFT生成器
        sft_generator = SFTDatasetGenerator(
            model_caller=model_caller,
            data_loader=data_loader,
            instruction_prompt=prompts['instruction'],
            input_prompt=prompts['input'],
            output_prompt=prompts['output']
        )
        
        # 创建优化的生成器
        optimized_generator = OptimizedDataGenerator(
            data_generator=sft_generator,
            checkpoint_dir=checkpoint_dir
        )
        
        print("组件初始化完成")
        
        # 测试检查点功能
        checkpoint_data = {
            'completed_samples': 1,
            'total_samples': 3,
            'mode': 'complete',
            'output_file': output_file
        }
        
        checkpoint_id = optimized_generator.save_checkpoint(checkpoint_data)
        print(f"保存检查点: {checkpoint_id}")
        
        # 加载检查点
        loaded_checkpoint = optimized_generator.load_checkpoint(checkpoint_id)
        print(f"加载检查点: {loaded_checkpoint['completed_samples']}/{loaded_checkpoint['total_samples']}")
        
        # 清理检查点
        optimized_generator.delete_checkpoint(checkpoint_id)
        print("检查点已删除")
        
        # 清理临时文件
        os.unlink(test_file)
        import shutil
        shutil.rmtree(output_dir)
        
        print("内存优化和断点续传测试完成\n")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        print("这可能是因为缺少某些依赖或配置\n")

# 测试配置文件
def test_config_files():
    print("=== 测试配置文件 ===")
    
    # 检查配置文件是否存在
    config_files = [
        "config/config.py",
        "config/prompt_config.py",
        "config/prompt_configs.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} 存在")
        else:
            print(f"❌ {config_file} 不存在")
    
    print("配置文件检查完成\n")

# 主测试函数
def main():
    print("开始测试新功能...\n")
    
    # 测试配置文件
    test_config_files()
    
    # 测试提示词版本管理
    try:
        test_prompt_version_management()
    except Exception as e:
        print(f"提示词版本管理测试失败: {str(e)}\n")
    
    # 测试内存优化和断点续传
    try:
        test_optimized_generator()
    except Exception as e:
        print(f"内存优化和断点续传测试失败: {str(e)}\n")
    
    print("所有测试完成！")

if __name__ == "__main__":
    main()