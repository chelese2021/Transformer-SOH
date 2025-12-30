@echo off
echo ============================================================
echo 安装GPU版本PyTorch到battery_gpu环境
echo ============================================================

echo.
echo [步骤1] 激活conda环境...
call conda activate battery_gpu

echo.
echo [步骤2] 安装PyTorch GPU版本...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [步骤3] 安装其他依赖...
pip install pandas scikit-learn matplotlib seaborn tqdm

echo.
echo [步骤4] 验证GPU...
python -c "import torch; print('='*60); print('GPU配置检查'); print('='*60); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU数量: {torch.cuda.device_count()}'); print('='*60)"

echo.
echo 安装完成！
pause
