@echo off
call "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
cd /d "D:\picolm\picolm"
echo __global__ void k() {} > test_min.cu
"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" -c -o test_min.obj test_min.cu
echo "nvcc exit=%errorlevel%"
del test_min.cu test_min.obj 2>nul
