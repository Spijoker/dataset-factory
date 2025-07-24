# -*- coding: utf-8 -*-
"""
文件上传管理模块
用于处理Streamlit文件上传和临时文件管理
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union
import streamlit as st
from datetime import datetime

class FileUploadManager:
    """
    文件上传管理器
    负责处理文件上传、临时文件存储和管理
    """
    
    def __init__(self, project_root: str = None):
        """
        初始化文件上传管理器
        
        Args:
            project_root: 项目根目录，如果为None则自动检测
        """
        if project_root is None:
            # 自动检测项目根目录
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent
        else:
            self.project_root = Path(project_root)
        
        # 创建TMP目录
        self.tmp_dir = self.project_root / "TMP"
        self.ensure_tmp_dir()
    
    def ensure_tmp_dir(self):
        """
        确保TMP目录存在
        """
        try:
            self.tmp_dir.mkdir(exist_ok=True)
            # 创建一个.gitignore文件，避免上传的临时文件被提交到git
            gitignore_path = self.tmp_dir / ".gitignore"
            if not gitignore_path.exists():
                with open(gitignore_path, 'w', encoding='utf-8') as f:
                    f.write("# 忽略所有上传的临时文件\n*\n!.gitignore\n")
        except Exception as e:
            st.error(f"创建TMP目录失败: {str(e)}")
    
    def get_tmp_dir_path(self) -> str:
        """
        获取TMP目录的绝对路径
        
        Returns:
            TMP目录的绝对路径字符串
        """
        return str(self.tmp_dir.absolute())
    
    def save_uploaded_file(self, uploaded_file, custom_filename: str = None) -> Optional[str]:
        """
        保存上传的文件到TMP目录
        
        Args:
            uploaded_file: Streamlit上传的文件对象
            custom_filename: 自定义文件名，如果为None则使用原文件名
            
        Returns:
            保存的文件路径，如果失败返回None
        """
        try:
            if uploaded_file is None:
                return None
            
            # 确定文件名
            if custom_filename:
                filename = custom_filename
                if not filename.endswith('.json'):
                    filename += '.json'
            else:
                filename = uploaded_file.name
            
            # 添加时间戳避免文件名冲突
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = os.path.splitext(filename)
            unique_filename = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
            
            # 保存文件
            file_path = self.tmp_dir / unique_filename
            
            # 验证是否为有效的JSON文件
            try:
                file_content = uploaded_file.read()
                if isinstance(file_content, bytes):
                    file_content = file_content.decode('utf-8')
                
                # 验证JSON格式
                json.loads(file_content)
                
                # 保存文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                
                return str(file_path.absolute())
                
            except json.JSONDecodeError as e:
                st.error(f"文件 {filename} 不是有效的JSON格式: {str(e)}")
                return None
            except UnicodeDecodeError as e:
                st.error(f"文件 {filename} 编码格式不支持: {str(e)}")
                return None
                
        except Exception as e:
            st.error(f"保存文件失败: {str(e)}")
            return None
    
    def save_uploaded_files(self, uploaded_files: List) -> List[str]:
        """
        批量保存上传的文件
        
        Args:
            uploaded_files: Streamlit上传的文件对象列表
            
        Returns:
            成功保存的文件路径列表
        """
        saved_files = []
        
        for uploaded_file in uploaded_files:
            file_path = self.save_uploaded_file(uploaded_file)
            if file_path:
                saved_files.append(file_path)
        
        return saved_files
    
    def list_tmp_files(self) -> List[str]:
        """
        列出TMP目录中的所有JSON文件
        
        Returns:
            JSON文件路径列表
        """
        try:
            json_files = list(self.tmp_dir.glob("*.json"))
            return [str(f.absolute()) for f in json_files]
        except Exception as e:
            st.error(f"列出临时文件失败: {str(e)}")
            return []
    
    def delete_tmp_file(self, file_path: str) -> bool:
        """
        删除指定的临时文件
        
        Args:
            file_path: 要删除的文件路径
            
        Returns:
            删除是否成功
        """
        try:
            file_path = Path(file_path)
            if file_path.exists() and file_path.parent == self.tmp_dir:
                file_path.unlink()
                return True
            return False
        except Exception as e:
            st.error(f"删除文件失败: {str(e)}")
            return False
    
    def clear_tmp_dir(self) -> bool:
        """
        清空TMP目录（保留.gitignore文件）
        
        Returns:
            清空是否成功
        """
        try:
            for file_path in self.tmp_dir.iterdir():
                if file_path.name != ".gitignore":
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
            return True
        except Exception as e:
            st.error(f"清空临时目录失败: {str(e)}")
            return False
    
    def create_folder(self, folder_name: str) -> bool:
        """
        在TMP目录中创建文件夹
        
        Args:
            folder_name: 文件夹名称
            
        Returns:
            创建是否成功
        """
        try:
            folder_path = self.tmp_dir / folder_name
            if folder_path.exists():
                st.warning(f"文件夹 '{folder_name}' 已存在")
                return False
            
            folder_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            st.error(f"创建文件夹失败: {str(e)}")
            return False
    
    def list_folders(self) -> List[str]:
        """
        列出TMP目录中的所有文件夹
        
        Returns:
            文件夹名称列表
        """
        try:
            folders = []
            for item in self.tmp_dir.iterdir():
                if item.is_dir() and item.name != ".git":
                    folders.append(item.name)
            return sorted(folders)
        except Exception as e:
            st.error(f"列出文件夹失败: {str(e)}")
            return []
    
    def move_file_to_folder(self, file_path: str, folder_name: str) -> bool:
        """
        将文件移动到指定文件夹
        
        Args:
            file_path: 源文件路径
            folder_name: 目标文件夹名称，空字符串表示根目录
            
        Returns:
            移动是否成功
        """
        try:
            source_path = Path(file_path)
            
            # 确定目标文件夹
            if folder_name and folder_name.strip():
                target_folder = self.tmp_dir / folder_name
                # 确保目标文件夹存在
                if not target_folder.exists():
                    target_folder.mkdir(parents=True, exist_ok=True)
            else:
                # 移动到根目录
                target_folder = self.tmp_dir
            
            # 移动文件
            target_path = target_folder / source_path.name
            if target_path.exists():
                st.warning(f"目标位置已存在同名文件: {source_path.name}")
                return False
            
            source_path.rename(target_path)
            return True
        except Exception as e:
            st.error(f"移动文件失败: {str(e)}")
            return False
    
    def list_files_in_folder(self, folder_name: str = None) -> List[str]:
        """
        列出指定文件夹中的JSON文件
        
        Args:
            folder_name: 文件夹名称，如果为None则列出根目录文件
            
        Returns:
            JSON文件路径列表
        """
        try:
            if folder_name:
                target_dir = self.tmp_dir / folder_name
            else:
                target_dir = self.tmp_dir
            
            if not target_dir.exists():
                return []
            
            json_files = list(target_dir.glob("*.json"))
            return [str(f.absolute()) for f in json_files]
        except Exception as e:
            st.error(f"列出文件失败: {str(e)}")
            return []
    
    def delete_folder(self, folder_name: str) -> bool:
        """
        删除指定文件夹及其内容
        
        Args:
            folder_name: 文件夹名称
            
        Returns:
            删除是否成功
        """
        try:
            folder_path = self.tmp_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)
                return True
            return False
        except Exception as e:
            st.error(f"删除文件夹失败: {str(e)}")
            return False
    
    def save_uploaded_file_to_folder(self, uploaded_file, folder_name: str = None, custom_filename: str = None) -> Optional[str]:
        """
        保存上传的文件到指定文件夹
        
        Args:
            uploaded_file: Streamlit上传的文件对象
            folder_name: 目标文件夹名称，如果为None则保存到根目录
            custom_filename: 自定义文件名，如果为None则使用原文件名
            
        Returns:
            保存的文件路径，如果失败返回None
        """
        try:
            if uploaded_file is None:
                return None
            
            # 确定保存目录
            if folder_name:
                save_dir = self.tmp_dir / folder_name
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = self.tmp_dir
            
            # 确定文件名
            if custom_filename:
                filename = custom_filename
                if not filename.endswith('.json'):
                    filename += '.json'
            else:
                filename = uploaded_file.name
            
            # 添加时间戳避免文件名冲突
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = os.path.splitext(filename)
            unique_filename = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
            
            # 保存文件
            file_path = save_dir / unique_filename
            
            # 验证是否为有效的JSON文件
            try:
                file_content = uploaded_file.read()
                if isinstance(file_content, bytes):
                    file_content = file_content.decode('utf-8')
                
                # 验证JSON格式
                json.loads(file_content)
                
                # 保存文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                
                return str(file_path.absolute())
                
            except json.JSONDecodeError as e:
                st.error(f"文件 {filename} 不是有效的JSON格式: {str(e)}")
                return None
            except UnicodeDecodeError as e:
                st.error(f"文件 {filename} 编码格式不支持: {str(e)}")
                return None
                
        except Exception as e:
            st.error(f"保存文件失败: {str(e)}")
            return None
    
    def get_file_info(self, file_path: str) -> dict:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含文件信息的字典
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {}
            
            stat = file_path.stat()
            
            # 尝试读取JSON文件获取数据量
            data_count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        data_count = len(data)
                    else:
                        data_count = 1
            except:
                pass
            
            return {
                'name': file_path.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'data_count': data_count
            }
        except Exception as e:
            return {'error': str(e)}

# 全局文件上传管理器实例
file_upload_manager = FileUploadManager()