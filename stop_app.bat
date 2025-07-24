@echo off
chcp 65001 >nul
echo 🛑 正在关闭数据集生成器...
echo ========================================

echo 📋 终止Streamlit进程...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *streamlit*" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 已终止Streamlit相关进程
) else (
    echo ℹ️  未找到Streamlit窗口进程
)

echo 🔌 终止端口8501上的进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501"') do (
    taskkill /F /PID %%a >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✅ 已终止端口8501上的进程 (PID: %%a)
    )
)

echo 🐍 终止包含app.py的Python进程...
wmic process where "name='python.exe' and CommandLine like '%%app.py%%'" delete >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 已终止app.py相关进程
) else (
    echo ℹ️  未找到app.py相关进程
)

echo 🧹 清理Streamlit缓存...
wmic process where "name='python.exe' and CommandLine like '%%streamlit%%'" delete >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ 已清理Streamlit进程
) else (
    echo ℹ️  未找到其他Streamlit进程
)

echo.
echo ========================================
echo 🎉 关闭操作完成！
echo 💡 如果浏览器页面仍然显示，请手动刷新或关闭
echo 💡 现在可以安全地重新启动应用: python run_app.py
echo.
pause