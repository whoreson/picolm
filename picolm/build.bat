@echo off
REM PicoLM Windows build script (MSVC)
REM
REM SIMD targets:
REM   build.bat          -- SSE2 baseline (safe for any x86-64)
REM   build.bat avx2     -- AVX2 (Haswell+ / Excavator+, fastest)
REM   build.bat avx      -- AVX  (Sandy Bridge+ / Bulldozer+)
REM   build.bat scalar   -- no SIMD (portable scalar fallback)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set SIMD_FLAG=
if /I "%1"=="avx2"   set SIMD_FLAG=/arch:AVX2
if /I "%1"=="avx"    set SIMD_FLAG=/arch:AVX
if /I "%1"=="scalar" set SIMD_FLAG=/d2archSSE42-

if "%SIMD_FLAG%"=="" (
    echo Building: SSE2 baseline
) else (
    echo Building: %1 ^(%SIMD_FLAG%^)
)

cl /O2 /W3 %SIMD_FLAG% /Fe:picolm.exe picolm.c model.c tensor.c quant.c tokenizer.c sampler.c grammar.c
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
) else (
    echo BUILD SUCCESS
)
