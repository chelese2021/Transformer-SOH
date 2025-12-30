@echo off
echo ============================================================
echo 使用GPU训练电池SOC/SOH预测模型
echo ============================================================
echo.
echo GPU信息:
call conda run -n battery_gpu python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
echo.
echo ============================================================
echo 开始训练...
echo ============================================================
echo.

cd /d E:\SOH
call conda run -n battery_gpu python train.py

echo.
echo ============================================================
echo 训练完成！
echo ============================================================
pause
