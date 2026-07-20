@echo off
REM PicoLM GPU build script for Windows (CUDA)
REM
REM Requires:
REM   - CUDA toolkit 12.4 (default install path)
REM   - Visual Studio 2022 Community (or BuildTools)
REM
REM Usage: build_gpu_win.bat
REM Produces: picolm.exe with GPU support linked in
REM
REM If building fails due to wrong VS path, edit the 'call vcvars...' line below.

echo PicoLM GPU Build (CUDA)

REM Try different Visual Studio paths
call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Could not find Visual Studio. Install VS2022 Community or BuildTools.
    exit /b 1
)

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo ERROR: nvcc not found at %%CUDA_PATH%%. Edit CUDA_PATH above.
    exit /b 1
)

echo Compiling GPU kernels with nvcc...
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++11 -DPICOLM_GPU=1 ^
    -Xcompiler "/EHsc /O2 /W3 /Zi" ^
    -c backend_gpu.hip -o backend_gpu_cu.obj ^
    -I"%CUDA_PATH%\include" -I.
if %ERRORLEVEL% neq 0 (
    echo ERROR: nvcc compilation failed.
    exit /b 1
)

echo Compiling host sources with cl (as C++ to link with nvcc output)...
cl /O2 /W3 /EHsc /DPICOLM_GPU=1 /I. /Fe:picolm_gpu.exe ^
    /Tp picolm.c model.c tensor.c quant.c tokenizer.c sampler.c grammar.c ^
    /Tp cJSON.c server.c csafetensors.c json.c safetensors.c ^
    backend_gpu_cu.obj ^
    "%CUDA_PATH%\lib\x64\cudart.lib"
if %ERRORLEVEL% neq 0 (
    echo ERROR: cl compilation failed.
    del backend_gpu_cu.obj
    exit /b 1
)

del backend_gpu_cu.obj
echo BUILD SUCCESS: picolm_gpu.exe
echo Run with: set PICOLM_GPU=1 ^& picolm_gpu.exe ^<model.gguf^> -p "hello"

