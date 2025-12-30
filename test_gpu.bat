@echo off
echo ============================================================
echo 测试GPU环境
echo ============================================================
echo.
call conda run -n battery_gpu python -c "import torch; print('='*60); print('GPU配置验证'); print('='*60); print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0)}'); print(f'GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB'); print('='*60); print(); print('测试GPU计算...'); x = torch.randn(1000, 1000).cuda(); y = torch.randn(1000, 1000).cuda(); z = torch.matmul(x, y); print('GPU计算正常！'); print('='*60)"
echo.
pause
