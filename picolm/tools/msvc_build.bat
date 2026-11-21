@echo off
rem ============================================================================
rem  tools/msvc_build.bat -- native MSVC build for PicoLM, driven by GNU make.
rem
rem  Invoked as:   make msvc          (CPU only, SIMD autodetected)
rem                make msvc-gpu      (adds CUDA)
rem                make msvc SIMD=avx2
rem
rem  Why a checked-in .bat instead of echoing one from the Makefile:
rem    1. cmd.exe quoting is native here. GNU make's $(shell) mangles quotes
rem       on Windows ("..." arrives as \"..."), so discovery cannot live in
rem       make. This is what made the old `hunger` target so fragile.
rem    2. It is debuggable: you can run tools\msvc_build.bat by hand from any
rem       CMD prompt / FAR Manager, with no make involved.
rem
rem  Why no hardcoded paths: VS is found with vswhere.exe (shipped by every
rem  VS 2017+ installer), the toolset version and VS root are derived from the
rem  cl.exe path, and the Windows SDK comes from what vcvars64.bat sets up.
rem  Nothing here assumes MSYS2, and nothing assumes a specific VS or CUDA.
rem
rem  Args (all via environment, set by make):
rem    MSVC_SIMD   native|avx512|avx2|avx|sse2   (default: native)
rem    MSVC_GPU    1 to compile the CUDA backend   (default: unset)
rem    MSVC_JOBS   parallel cl.exe invocations     (default: 1, cl is +/MP)
rem  Exit: 0 ok, non-zero on any failure.
rem ============================================================================
setlocal EnableDelayedExpansion

rem ---- argument parsing ------------------------------------------------------
rem Args (from `make`: --simd=<tier> [--gpu]; or set the env vars by hand):
rem   --simd:Tier      avx512|avx2|avx|sse2|native  (default native=probe)
rem   --gpu                                  compile the CUDA backend
rem Arguments rather than env vars because GNU make's recipe must work under
rem both sh.exe and cmd.exe, and no single "set VAR=x cmd..." spelling is valid
rem in both. Arguments are shell-neutral.
rem Plain goto flow, NO parenthesised blocks: `shift` inside a (...) block does
rem not take effect until the block closes, so a block-based parser either loops
rem forever on the same argument or drops it. Each test therefore fails its way
rem to the next label and only ever shifts at top level.
rem
rem Separator is ":" not "=" -- when GNU make runs the recipe through sh.exe,
rem sh word-splits an argument containing "=" at the "=", so "--simd=avx2"
rem arrives as two arguments ("--simd", "avx2"). It does not touch ":". 
set "MSVC_SIMD="
set "MSVC_GPU="
:parse_args
if "%~1"=="" goto args_done
set "ARG=%~1"
if /i "%ARG:~0,7%"=="--simd:" goto arg_simd
if /i "%ARG%"=="--gpu" goto arg_gpu
if /i "%ARG%"=="--selftest" goto arg_selftest
if /i "%ARG%"=="--print-tier" goto arg_print_tier
if /i "%ARG%"=="--viz" goto arg_viz
echo WARN: ignoring unknown argument "%ARG%"
shift
goto parse_args
:arg_simd
set "MSVC_SIMD=%ARG:~7%"
shift
goto parse_args
:arg_gpu
set "MSVC_GPU=1"
shift
goto parse_args
:arg_selftest
set "MSVC_SELFTEST=1"
shift
goto parse_args
:arg_print_tier
set "MSVC_PRINT_TIER=1"
shift
goto parse_args
:arg_viz
set "MSVC_VIZ=1"
shift
goto parse_args
:args_done

rem ---- sane defaults --------------------------------------------------------
if not defined MSVC_SIMD set "MSVC_SIMD=native"
if not defined MSVC_JOBS set "MSVC_JOBS=1"

rem Working directory: the repository root. make runs us from there, so %CD%
rem is authoritative. We deliberately DO NOT derive it from %~dp0: when the
rem recipe is executed through MSYS2 sh.exe the "tools/" component is not
rem always preserved in the path handed to cmd, which made %~dp0 come back as
rem the repo root and sent the build one directory too high (its CPU probe was
rem then not found, it silently fell back to SSE2, and cl.exe failed).
rem If the caller ran us by path from inside tools\, %~dp0 is trustworthy and
rem we cd to its parent. Otherwise we are already in the repo root; trust %CD%.
rem
rem This test must not shell out. `find` is the Unix find.exe on any machine
rem that has MSYS2/MinGW on PATH, so `echo ... | find "\tools\"` tries to open
rem a path called "\tools\" and fails; `findstr` is Windows-only. Comparing the
rem tail of %~dp0 in pure batch has no external dependency at all.
set "DP0=%~dp0"
if /i "!DP0:~-7!"=="\tools\" pushd "!DP0!.."
if not exist "tools\msvc_cpu_probe.c" goto bad_cwd
goto cwd_ok
:bad_cwd
echo ERROR: not in the repository root - tools\msvc_cpu_probe.c not found.
echo        cwd: %CD%
exit /b 1
:cwd_ok

rem In --print-tier mode stdout must carry exactly one line (the tier), because
rem CI reads it with `for /f`. Everything else goes to stderr.
if defined MSVC_PRINT_TIER goto banner_done
echo === PicoLM MSVC build ===
echo   cwd:  %CD%
echo   SIMD: %MSVC_SIMD%
if defined MSVC_GPU (echo   GPU:  yes) else (echo   GPU:  no)
:banner_done

rem ---- 1. locate Visual Studio --------------------------------------------
rem vswhere.exe is installed with the VS Installer and is the documented,
rem version-independent way to find a VS install. Try the standard locations,
rem then PATH as a last resort.
rem NOTE: %ProgramFiles(x86)% contains a ")" -- expanding it directly inside an
rem `if exist (...)` breaks batch parsing. Copy it to a plain variable first.
set "PF86=%ProgramFiles(x86)%"
if not defined PF86 set "PF86=%ProgramW6432%"
if not defined PF86 if defined SystemDrive set "PF86=%SystemDrive%\Program Files (x86)"
if not defined PF86 set "PF86=C:\Program Files (x86)"
set "PF64=%ProgramFiles%"
if not defined PF64 set "PF64=%ProgramW6432%"
if not defined PF64 if defined SystemDrive set "PF64=%SystemDrive%\Program Files"
if not defined PF64 set "PF64=C:\Program Files"

set "VSWHERE="
if exist "%PF86%\Microsoft Visual Studio\Installer\vswhere.exe" set "VSWHERE=%PF86%\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined VSWHERE if exist "%PF64%\Microsoft Visual Studio\Installer\vswhere.exe" set "VSWHERE=%PF64%\Microsoft Visual Studio\Installer\vswhere.exe"
if not defined VSWHERE for %%i in (vswhere.exe) do if not "%%~$PATH:i"=="" set "VSWHERE=%%~$PATH:i"
if not defined VSWHERE (
    echo ERROR: vswhere.exe not found.
    echo        Install Visual Studio 2017+ or Build Tools with the
    echo        "Desktop development with C++" workload
    popd 2>nul & exit /b 1
)

rem -products * also finds Build Tools, not just the full IDE.
set "CL_EXE="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe`) do set "CL_EXE=%%i"
if not defined CL_EXE (
    echo ERROR: no x64 MSVC toolset found via vswhere.
    echo        Add the "MSVC v143 x64/x86 build tools" component.
    popd 2>nul & exit /b 1
)

rem Derive the VS root from cl.exe:  <VS>\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe
rem is six levels below <VS>, so walk up six times. No version pinning.
for %%i in ("%CL_EXE%") do set "CL_DIR=%%~dpi"
set "VS_ROOT=%CL_DIR%..\..\..\..\..\.."
for %%i in ("%VS_ROOT%") do set "VS_ROOT=%%~fi"
set "VCVARS=%VS_ROOT%\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo ERROR: vcvars64.bat not found at "%VCVARS%"
    popd 2>nul & exit /b 1
)
if not defined MSVC_PRINT_TIER echo   cl:   %CL_EXE%
if not defined MSVC_PRINT_TIER echo   vars: %VCVARS%

rem link.exe and cl.exe need a writable TEMP. If TEMP/TMP are unset or point
rem somewhere unwritable (link.exe then tries C:\Windows and fails with
rem LNK1104 on lnk{GUID}.tmp), fall back to a directory we create here.
if not defined TEMP set "TEMP=%CD%\.msvc_tmp"
if not defined TMP set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%" 2>nul

rem ---- 2. enter the MSVC environment --------------------------------------
rem vcvars64 sets INCLUDE / LIB / LIBPATH for the detected toolset + SDK.
rem Doing that here (rather than hand-assembling those paths as the old
rem hunger-msys-direct target did) is what makes this survive VS upgrades.
call "%VCVARS%" >nul
if errorlevel 1 (echo ERROR: vcvars64.bat failed & popd 2>nul & exit /b 1)

rem ---- 3. detect the CPU's SIMD tier --------------------------------------
rem A compiled probe, not /proc/cpuinfo or WMI: those do not exist outside
rem MSYS2, and /proc/cpuinfo cannot report AVX-512 reliably. Cached in
rem .msvc_cpu so the cost is one tiny compile, once.
if not defined MSVC_SIMD set "MSVC_SIMD=native"
if /i "%MSVC_SIMD%"=="native" (
    if exist ".msvc_cpu" (
        set /p MSVC_SIMD=<.msvc_cpu
        if not defined MSVC_PRINT_TIER echo   cpu:  cached
    ) else (
        if not defined MSVC_PRINT_TIER echo   cpu:  probing...
        cl /nologo /O2 /Fe:.msvc_probe.exe tools\msvc_cpu_probe.c >nul 2>&1
        if not exist ".msvc_probe.exe" (
            echo WARN: CPU probe failed to compile; assuming SSE2
            set "MSVC_SIMD=sse2"
        ) else (
            for /f "usebackq delims=" %%i in (`".msvc_probe.exe"`) do set "PROBE=%%i"
            del /q ".msvc_probe.exe" ".msvc_probe.obj" 2>nul
            rem Pick the highest tier the CPU actually has AND cl.exe accepts.
            set "MSVC_SIMD=sse2"
            rem Use the real Windows find.exe by absolute path. When this
            rem script is started from an MSYS2/Cygwin shell -- which `make
            rem msvc` does, and which the notes support -- PATH holds their
            rem POSIX `find`, whose arguments are paths, not search strings.
            rem Every token then "fails to match", the tier silently drops to
            rem sse2, and the build succeeds with a scalar binary. Same class
            rem of bug the .msvc_cpu comment warns about; this removes the
            rem dependency on PATH entirely.
            set "FIND=!SystemRoot!\System32\find.exe"
            echo !PROBE! | "!FIND!" "avx " >nul && set "MSVC_SIMD=avx"
            echo !PROBE! | "!FIND!" "avx2 " >nul && set "MSVC_SIMD=avx2"
            echo !PROBE! | "!FIND!" "avx512f " >nul && set "MSVC_SIMD=avx512"
            rem AVX-512 VNNI is reported separately: it needs its own -D because
            rem MSVC never predefines __AVX512VNNI__, so quant.h cannot derive it.
            echo !PROBE! | "!FIND!" "vnni " >nul && echo vnni> ".msvc_vnni"
            echo !MSVC_SIMD!> ".msvc_cpu"
            if not defined MSVC_PRINT_TIER echo   cpu:  !MSVC_SIMD!
        )
    )
)

rem ---- 4. translate the tier into an /arch: flag --------------------------
rem Verification matters: older cl.exe versions accept /arch:AVX2 but not
rem /arch:AVX512, and the toolset may lack AVX-512 intrinsics. Test-compile
rem rather than assume -- a wrong assumption here silently produces a slower
rem or, worse, a differently-dispatched binary.
set "MSVC_ARCH="
if /i "%MSVC_SIMD%"=="avx512" set "MSVC_ARCH=/arch:AVX512"
if /i "%MSVC_SIMD%"=="avx2"   set "MSVC_ARCH=/arch:AVX2"
if /i "%MSVC_SIMD%"=="avx"    set "MSVC_ARCH=/arch:AVX"
if /i "%MSVC_SIMD%"=="sse2"   set "MSVC_ARCH=/arch:SSE2"
if not defined MSVC_ARCH (
    echo ERROR: unknown MSVC_SIMD=%MSVC_SIMD% - use avx512/avx2/avx/sse2
    popd 2>nul & exit /b 1
)

rem ---- 4b. AVX-512 VNNI ---------------------------------------------------
rem MSVC defines no __AVX512VNNI__ for /arch:AVX512, so quant.h's PICOLM_VNNI
rem gate (which needs __AVX512VNNI__ && __AVX512VL__) can never fire under
rem cl.exe and every VNNI kernel compiles out -- the build silently runs the
rem AVX2 fallbacks (sgemm_iq6_k_q8_k_r8t instead of r8v,
rem sgemm_q8_0_r8_avx2t_q8_0 instead of the VNNI variant) on a CPU that has
rem VNNI. Verified on VS2022 14.44: /arch:AVX512 emits vpdpbusd, and the CPU
rem probe reports the feature bit; only the macro was missing.
rem Gated on the probe (not on the /arch: flag) so a Skylake-X or a VM
rem without ZMM state never executes vpdpbusd.
set "MSVC_VNNI_DEF="
if not exist ".msvc_vnni" goto vnni_done
set /p MSVC_VNNI_TMP=<.msvc_vnni
if /i not "%MSVC_SIMD%"=="avx512" goto vnni_done
set "MSVC_VNNI_DEF=/D PICOLM_VNNI=1"
if not defined MSVC_PRINT_TIER echo   vnni: yes (/D PICOLM_VNNI=1)
:vnni_done

rem Use a real translation unit for the test. "echo . > f.c" writes a stray
rem "." line and cl.exe rejects it as a syntax error, which made this check
rem always fail and silently cap every build at the fallback tier.
copy /y nul ".msvc_archtest.c" >nul 2>&1
>> ".msvc_archtest.c" echo void picolm_archtest^(void^) ^{^}
cl /nologo /c %MSVC_ARCH% ".msvc_archtest.c" /Fo:nul >nul 2>&1
if errorlevel 1 (
    echo WARN: cl.exe rejected %MSVC_ARCH%; falling back.
    del /q ".msvc_archtest.c" ".msvc_archtest.obj" 2>nul

rem ---- selftest --------------------------------------------------------
rem Placement matters: the CPU-side checks need the tier resolved but nothing
rem compiled, so they run here and `make msvc-selftest` stays cheap. When --gpu
rem is also given the nvcc/cudart checks need step 5a to have run, so we defer
rem the whole selftest until after CUDA discovery in that case.
if not defined MSVC_SELFTEST goto st_skip
if defined MSVC_GPU goto st_skip
call :selftest_early
if errorlevel 1 goto fail
goto selftest_only
:st_skip
    if /i "%MSVC_SIMD%"=="avx512" (
        set "MSVC_SIMD=avx2" & set "MSVC_ARCH=/arch:AVX2"
    ) else if /i "%MSVC_SIMD%"=="avx2" (
        set "MSVC_SIMD=avx" & set "MSVC_ARCH=/arch:AVX"
    ) else (
        set "MSVC_SIMD=sse2" & set "MSVC_ARCH=/arch:SSE2"
    )
    if not defined MSVC_PRINT_TIER echo   tier: !MSVC_SIMD! -fallback-
) else (
    if not defined MSVC_PRINT_TIER echo   tier: %MSVC_SIMD%
)
del /q ".msvc_archtest.c" ".msvc_archtest.obj" 2>nul

rem --print-tier: emit the resolved tier and stop. Used by CI to pick a tier
rem the runner CPU can actually execute, without duplicating the
rem vswhere -> vcvars -> probe -> /arch-verify chain in the workflow.
if not defined MSVC_PRINT_TIER goto pt_skip
echo !MSVC_SIMD!
popd 2>nul
exit /b 0
:pt_skip

rem ---- 5. compile --------------------------------------------------------
rem No -D PICOLM_AVX2/AVX512/AVX/FMA/SSE2 here on purpose: quant.h derives them
rem from __AVX2__/__AVX__/__AVX512F__ (plus the MSVC __FMA__/__F16C__ shim), and
rem its /arch consistency guard #errors if a -D disagrees with the real /arch:.
rem Keeping the cascade in one place is what stops targets from drifting apart.
set "MSVC_WD=/wd4996 /wd4244 /wd4267 /wd4101 /wd4102 /wd4090 /wd4018 /wd4477"
set "MSVC_CF=/nologo /O2 /W3 %MSVC_WD% %MSVC_ARCH% /D _CRT_SECURE_NO_WARNINGS /I. /openmp /MP /c"
set "MSVC_CF=%MSVC_CF% /D PICOLM_SERVER=1"
if defined MSVC_VNNI_DEF set "MSVC_CF=!MSVC_CF! !MSVC_VNNI_DEF!"

set "MSVC_CSRCS="
for %%f in (picolm.c model_core.c model_recurrent_state.c model_attention.c model_moe.c model_gemma3n.c model_glm5.c model_ssm.c model_kv_cache.c model_gguf.c steer.c tensor.c type_dispatch.c quant.c sgemm.c sgemm_q0_neon_fast.c sgemm_q2_k_r4_neon.c sgemm_iq4nl_neon_fast.c sgemm_k_q8k_neon_fast.c sgemm_q2_k_neon.c sgemm_k_q8k_neon.c sgemm_q4_0x8.c sgemm_q4i_0x8.c sgemm_q4_0_r8.c sgemm_q8_0_r8.c sgemm_q8_k_r8.c sgemm_q8_k_r8_neon.c sgemm_iq2_k.c sgemm_iq2_k_r4.c sgemm_iq3_k.c sgemm_iq3_k_r4.c sgemm_iq4_k.c sgemm_iq4_nl_r4.c sgemm_q4_k_r4.c sgemm_q6_k_r4.c sgemm_iq6_k.c sgemm_iq4_xs.c tokenizer.c qwen_tokenize.c sampler.c grammar.c cJSON.c server.c csafetensors.c json.c safetensors.c) do (
    set "MSVC_CSRCS=!MSVC_CSRCS! %%f"
)

rem viz.c is opt-in, matching `make viz` on the Unix side: it adds -DPICOLM_VIZ
rem to every translation unit that has viz_push_layer calls, plus viz.c itself.
if defined MSVC_VIZ (
    set "MSVC_CSRCS=!MSVC_CSRCS! viz.c"
    set "MSVC_CF=!MSVC_CF! /D PICOLM_VIZ=1"
    if not defined MSVC_PRINT_TIER echo   viz:  enabled ^(embedded VNC viewer^)
)

set "MSVC_LIBS=ws2_32.lib advapi32.lib"
set "MSVC_CUOBJS="

rem ---- 5a. CUDA, when requested -------------------------------------------
rem goto flow, no (...) blocks. Every variable below must be read with %VAR% at
rem top level (parse-time) or !VAR! where it was just assigned inside a loop --
rem mixing the two inside a block is what made an earlier version print an empty
rem nvcc path and then fail to extend INCLUDE, so cl.exe could not find
rem cuda_profiler_api.h.
if not defined MSVC_GPU goto cuda_done

set "MSVC_CF=!MSVC_CF! /D PICOLM_GPU=1 /D PICOLM_CUDA=1"

rem Prefer nvcc on PATH; otherwise take the newest toolkit under Program Files.
rem Directory names sort as v12.2 < v12.4, so iterating and keeping the last
rem match yields the newest installed version.
set "NVCC="
for %%i in (nvcc.exe) do if not "%%~$PATH:i"=="" set "NVCC=%%~$PATH:i"
if defined NVCC goto cuda_have_nvcc
for /d %%d in ("%PF64%\NVIDIA GPU Computing Toolkit\CUDA\v*") do if exist "%%d\bin\nvcc.exe" set "NVCC=%%d\bin\nvcc.exe"
if not defined NVCC goto cuda_missing
:cuda_have_nvcc

for %%i in ("!NVCC!") do set "NVCC_DIR=%%~dpi"
for %%i in ("!NVCC_DIR!..") do set "CUDA_ROOT=%%~fi"
if not defined MSVC_PRINT_TIER echo   nvcc: !NVCC!
if not defined MSVC_PRINT_TIER echo   cuda: !CUDA_ROOT!

rem Extend the vcvars environment with CUDA's headers and import libs, and put
rem the CUDA bin dir first so the runtime DLLs resolve when tests run.
set "INCLUDE=!INCLUDE!;!CUDA_ROOT!\include"
set "LIB=!LIB!;!CUDA_ROOT!\lib\x64"
set "PATH=!NVCC_DIR!;!PATH!"
set "MSVC_LIBS=!MSVC_LIBS! "!CUDA_ROOT!\lib\x64\cudart.lib""

rem nvcc needs the PICOLM_* SIMD tier EXPLICITLY. -Xcompiler /arch:... only
rem reaches the device-side cl invocation; the host pass that #includes quant.h
rem sees neither /arch: nor __AVX512F__, so without these defines quant.h
rem correctly concludes "no SIMD" and every .cu host function falls back to
rem scalar code -- silent, and slow. This is the one place the cascade is
rem spelled out by hand, because nvcc cannot be told "/arch: and derive".
rem It is kept in step with MSVC_SIMD by the case below.
set "NVCC_SIMD_DEF="
if /i "%MSVC_SIMD%"=="avx512" set "NVCC_SIMD_DEF=-DPICOLM_AVX512=1 -DPICOLM_AVX2=1 -DPICOLM_AVX=1 -DPICOLM_SSE3=1 -DPICOLM_SSE2=1 -DPICOLM_FMA=1"
if /i "%MSVC_SIMD%"=="avx2"   set "NVCC_SIMD_DEF=-DPICOLM_AVX2=1 -DPICOLM_AVX=1 -DPICOLM_SSE3=1 -DPICOLM_SSE2=1 -DPICOLM_FMA=1"
if /i "%MSVC_SIMD%"=="avx"    set "NVCC_SIMD_DEF=-DPICOLM_AVX=1 -DPICOLM_SSE3=1 -DPICOLM_SSE2=1 -DPICOLM_FMA=1"
if /i "%MSVC_SIMD%"=="sse2"   set "NVCC_SIMD_DEF=-DPICOLM_SSE3=1 -DPICOLM_SSE2=1"

rem /arch is also forwarded to nvcc's inner cl.exe via -Xcompiler so the device
rem side matches the C side.
rem PICOLM_NVCC_EXTRA: extra nvcc flags, e.g. -DPICOLM_ABL_NO_BDECODE for the
rem prefill ablation experiments. Kept as an env var rather than a Makefile
rem option because it is for throwaway measurements, not for shipping builds.
set "NVCC_EXTRA=%PICOLM_NVCC_EXTRA%"
set "NVCC_FLAGS=-O3 --fmad=false -std=c++14 %NVCC_EXTRA% %NVCC_SIMD_DEF% -DPICOLM_GPU=1 -DPICOLM_CUDA=1 -DPICOLM_SSM_WARP_KERNEL_VALIDATED -DPICOLM_SSM_CHUNKED_GPU_VALIDATED -ccbin "!CL_EXE!" -Xcompiler "/EHsc /O2 /W3 /std:c++14 !MSVC_ARCH! /wd4244 /wd4267 /wd4101 /wd4102" -gencode arch=compute_89,code=sm_89 -I. -c -o"

rem --selftest only inspects the environment; it must not compile the .cu files
rem (that is the slow part of this target). Skip straight to the checks.
if defined MSVC_SELFTEST goto cuda_done

set "MSVC_CUOBJS="
for %%f in (backend_gpu_kernels.cu backend_gpu_host_core.cu backend_gpu_host_ssm.cu backend_gpu_host_misc.cu) do call :cuda_one "%%f"
if errorlevel 1 goto fail
goto cuda_done

:cuda_one
echo [nvcc] %~1
"!NVCC!" !NVCC_FLAGS! "%~n1_win.obj" "%~1"
if errorlevel 1 (
    echo ERROR: nvcc failed on %~1
    exit /b 1
)
set "MSVC_CUOBJS=!MSVC_CUOBJS! %~n1_win.obj"
exit /b 0

:cuda_missing
echo ERROR: --gpu requested but nvcc.exe was not found.
echo        Looked on PATH and in "%PF64%\NVIDIA GPU Computing Toolkit\CUDA\v*".
echo        Install the CUDA Toolkit, or add its bin directory to PATH.
goto fail

:cuda_done

rem Deferred selftest for --gpu: nvcc/cudart are known now.
if not defined MSVC_SELFTEST goto st_skip2
call :selftest_early
if errorlevel 1 goto fail
goto selftest_only
:st_skip2


echo [cl] compiling C sources (%MSVC_SIMD%)...
cl %MSVC_CF% !MSVC_CSRCS!
if errorlevel 1 (echo ERROR: cl.exe failed & popd 2>nul & exit /b 1)

rem ---- 6. link ------------------------------------------------------------
set "MSVC_COBJS="
for %%f in (!MSVC_CSRCS!) do set "MSVC_COBJS=!MSVC_COBJS! %%~nf.obj"

echo [link] picolm.exe
link /nologo /OUT:picolm.exe !MSVC_COBJS! !MSVC_CUOBJS! !MSVC_LIBS!
if errorlevel 1 (echo ERROR: link.exe failed & popd 2>nul & exit /b 1)

if defined MSVC_CUOBJS del /q !MSVC_CUOBJS! 2>nul

rem Copy the CUDA runtime next to the exe. Without this, running picolm.exe
rem needs the CUDA bin directory on PATH or Windows fails with
rem STATUS_DLL_NOT_FOUND (0xC0000135 / -1073741515) before main() runs --
rem which is a confusing way to find out. Only cudart is needed for the
rem current backend; cublas is copied when present for future kernels.
if defined MSVC_GPU (
    copy /y "!CUDA_ROOT!\bin\cudart64_*.dll" . >nul 2>&1
)

goto done

:fail
popd 2>nul
exit /b 1

:done

if defined MSVC_GPU (echo === OK: picolm.exe built, SIMD=%MSVC_SIMD%, GPU=enabled ===) else (echo === OK: picolm.exe built, SIMD=%MSVC_SIMD%, GPU=disabled ===)
popd 2>nul
exit /b 0

rem --selftest by itself stops after the checks; combined with a real
rem invocation it validates first and then builds.
:selftest_only
popd 2>nul
exit /b 0

rem ============================================================================
rem  --selftest : assert the environment is what the build assumes.
rem
rem  Every failure mode this catches was, at some point, SILENT: the build
rem  "succeeded" while quietly doing the wrong thing.
rem
rem    1. Wrong working directory. If tools\msvc_cpu_probe.c is not reachable,
rem       the probe fails to compile, the tier silently becomes SSE2, and the
rem       build still succeeds -- just far slower than asked for.
rem    2. cl.exe not found / vcvars not run. Usually a clear error, but the
rem       symptom can be a mystery "command not found" much later.
rem    3. The SIMD tier is not what was requested. The probe must agree with
rem       what cl.exe will accept; a mismatch means an explicit SIMD= was
rem       ignored, which is how the V-tile layout bug shipped.
rem    4. --gpu requested but nvcc missing, or cudart not usable at run time.
rem
rem  Exit 0 = all checks passed, 1 = at least one failed.
rem ============================================================================
:selftest_early
echo === PicoLM MSVC selftest ===

set "ST_FAIL=0"

rem --- 1. working directory ------------------------------------------------
echo   cwd:  %CD%
if not exist "tools\msvc_cpu_probe.c" goto st_cwd_bad
echo   [ok] repository root detected ^(tools\msvc_cpu_probe.c present^)
goto st_cl

:st_cwd_bad
echo   [FAIL] tools\msvc_cpu_probe.c not found -- wrong working directory.
echo          Without it the CPU probe cannot compile, the SIMD tier silently
echo          drops to SSE2 and the build still "succeeds".
set "ST_FAIL=1"

rem --- 2. compiler ---------------------------------------------------------
:st_cl
if not exist "%CL_EXE%" goto st_cl_bad
echo   [ok] cl.exe found
echo          %CL_EXE%
"%CL_EXE%" /? >nul 2>&1
if errorlevel 1 goto st_cl_bad
echo   [ok] cl.exe runs
goto st_arch

:st_cl_bad
echo   [FAIL] cl.exe missing or not runnable
set "ST_FAIL=1"

rem --- 3. SIMD tier --------------------------------------------------------
:st_arch
echo   [ok] tier resolved: %MSVC_SIMD% ^(%MSVC_ARCH%^)
rem Compile a real translation unit with the chosen /arch. "echo . > f.c"
rem writes a stray "." line that cl.exe rejects, which would make this check
rem always fail -- a trap already hit once in this file.
copy /y nul ".msvc_st.c" >nul 2>&1
>> ".msvc_st.c" echo void picolm_selftest^(void^) ^{^}
"%CL_EXE%" /nologo /c %MSVC_ARCH% ".msvc_st.c" /Fo:nul >nul 2>&1
if errorlevel 1 goto st_arch_bad
echo   [ok] cl.exe accepts %MSVC_ARCH%
del /q ".msvc_st.c" 2>nul
goto st_gpu

:st_arch_bad
echo   [FAIL] cl.exe rejected %MSVC_ARCH% -- the tier is not usable.
del /q ".msvc_st.c" 2>nul
set "ST_FAIL=1"

rem --- 4. CUDA -------------------------------------------------------------
:st_gpu
if not defined MSVC_GPU goto st_finish
if not defined NVCC goto st_gpu_bad
echo   [ok] nvcc found
echo          %NVCC%
if not exist "%CUDA_ROOT%\lib\x64\cudart.lib" goto st_gpu_bad
echo   [ok] cudart.lib present
if exist "cudart64_*.dll" goto st_gpu_dll
echo   [warn] cudart64_*.dll not copied next to picolm.exe yet
echo          run an actual build; it copies the DLL so the exe runs without
echo          the CUDA bin directory on PATH
goto st_finish
:st_gpu_dll
echo   [ok] cudart runtime DLL present next to the executable
goto st_finish

:st_gpu_bad
echo   [FAIL] --gpu requested but nvcc/cudart not found
set "ST_FAIL=1"

rem --- result --------------------------------------------------------------
:st_finish
if "%ST_FAIL%"=="0" goto st_pass
echo === selftest FAILED ===
exit /b 1
:st_pass
echo === selftest passed ===
exit /b 0
