@echo off
REM Must be run from Windows CMD, NOT from MSYS2 bash
REM Usage: cd /d D:\picolm\picolm && build_gpu.bat

call "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"

REM Restore user PATH (vcvars64.bat may have clobbered it)
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"

cd /d "D:\picolm\picolm"

REM Clean
del /q picolm.exe backend_gpu_kernels_win.obj backend_gpu_host_core_win.obj backend_gpu_host_ssm_win.obj backend_gpu_host_misc_win.obj 2>nul
del /q picolm.obj model_core.obj model_attention.obj model_moe.obj model_gemma3n.obj model_ssm.obj model_kv_cache.obj model_gguf.obj tensor.obj quant.obj sgemm.obj tokenizer.obj qwen_tokenize.obj sampler.obj grammar.obj cJSON.obj server.obj csafetensors.obj json.obj safetensors.obj 2>nul

echo === Compiling CUDA kernels ===
"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin cl -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=sm_89 -c -o backend_gpu_kernels_win.obj backend_gpu_kernels.cu -I.
if errorlevel 1 echo FAILED: backend_gpu_kernels.cu & exit /b 1

"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin cl -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=sm_89 -c -o backend_gpu_host_core_win.obj backend_gpu_host_core.cu -I.
if errorlevel 1 echo FAILED: backend_gpu_host_core.cu & exit /b 1

"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin cl -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=sm_89 -c -o backend_gpu_host_ssm_win.obj backend_gpu_host_ssm.cu -I.
if errorlevel 1 echo FAILED: backend_gpu_host_ssm.cu & exit /b 1

"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" -O3 --fmad=false -std=c++14 -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin cl -Xcompiler "/EHsc /O2 /W3 /std:c++14 /arch:AVX512 /wd4244 /wd4267 /wd4101 /wd4102 /DPICOLM_AVX512=1 /DPICOLM_AVX2=1 /DPICOLM_AVX=1 /DPICOLM_SSE3=1 /DPICOLM_SSE2=1 /DPICOLM_FMA=1" -gencode arch=compute_89,code=sm_89 -c -o backend_gpu_host_misc_win.obj backend_gpu_host_misc.cu -I.
if errorlevel 1 echo FAILED: backend_gpu_host_misc.cu & exit /b 1

echo === Compiling C sources ===
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 picolm.c
if errorlevel 1 echo FAILED: picolm.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 model_core.c
if errorlevel 1 echo FAILED: model_core.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 model_attention.c
if errorlevel 1 echo FAILED: model_attention.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 model_moe.c
if errorlevel 1 echo FAILED: model_moe.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 model_gemma3n.c
if errorlevel 1 echo FAILED: model_gemma3n.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 model_ssm.c
if errorlevel 1 echo FAILED: model_ssm.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 model_kv_cache.c
if errorlevel 1 echo FAILED: model_kv_cache.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 model_gguf.c
if errorlevel 1 echo FAILED: model_gguf.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 tensor.c
if errorlevel 1 echo FAILED: tensor.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 quant.c
if errorlevel 1 echo FAILED: quant.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 sgemm.c
if errorlevel 1 echo FAILED: sgemm.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 tokenizer.c
if errorlevel 1 echo FAILED: tokenizer.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 qwen_tokenize.c
if errorlevel 1 echo FAILED: qwen_tokenize.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 sampler.c
if errorlevel 1 echo FAILED: sampler.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 grammar.c
if errorlevel 1 echo FAILED: grammar.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 cJSON.c
if errorlevel 1 echo FAILED: cJSON.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 server.c
if errorlevel 1 echo FAILED: server.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 csafetensors.c
if errorlevel 1 echo FAILED: csafetensors.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 json.c
if errorlevel 1 echo FAILED: json.c & exit /b 1
cl.exe /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477 /arch:AVX512 /D PICOLM_GPU=1 /D PICOLM_CUDA=1 /D _CRT_SECURE_NO_WARNINGS /I. /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" /openmp /c /D PICOLM_AVX512=1 /D PICOLM_AVX2=1 /D PICOLM_AVX=1 /D PICOLM_SSE3=1 /D PICOLM_SSE2=1 /D PICOLM_FMA=1 safetensors.c
if errorlevel 1 echo FAILED: safetensors.c & exit /b 1

echo === Linking ===
link.exe /OUT:picolm.exe picolm.obj model_core.obj model_attention.obj model_moe.obj model_gemma3n.obj model_ssm.obj model_kv_cache.obj model_gguf.obj tensor.obj quant.obj sgemm.obj tokenizer.obj qwen_tokenize.obj sampler.obj grammar.obj cJSON.obj server.obj csafetensors.obj json.obj safetensors.obj backend_gpu_kernels_win.obj backend_gpu_host_core_win.obj backend_gpu_host_ssm_win.obj backend_gpu_host_misc_win.obj "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudart.lib" ws2_32.lib advapi32.lib
if errorlevel 1 echo FAILED: link & exit /b 1

echo === Cleaning CUDA intermediates ===
del /q backend_gpu_kernels_win.obj backend_gpu_host_core_win.obj backend_gpu_host_ssm_win.obj backend_gpu_host_misc_win.obj 2>nul

if exist picolm.exe (echo BUILD SUCCESSFUL) else (echo BUILD FAILED)
