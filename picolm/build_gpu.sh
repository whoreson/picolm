#!/usr/bin/env bash
cd /d/picolm/picolm
export MINGW_PREFIX=/ucrt64
export MINGW_CHOST=x86_64-w64-mingw32
export MINGW_PACKAGE_PREFIX=mingw-w64-ucrt-x86_64
export MSYSTEM_CARCH=x86_64
export MSYSTEM_CHOST=x86_64-w64-mingw32
export MSYSTEM_PREFIX=/ucrt64
export MSYSTEM=UCRT64
export ORIGINAL_TEMP=
export ORIGINAL_TMP=
export TEMP=/tmp
export TMP=/tmp
export XDG_DATA_DIRS=/ucrt64/share/:/usr/local/share/:/usr/share/
export CONFIG_SITE=/etc/config.site
export ACLOCAL_PATH=/ucrt64/share/aclocal:/usr/share/aclocal
export MANPATH=/ucrt64/local/man:/ucrt64/share/man:/usr/local/man:/usr/share/man:/usr/man:/share/man
export INFOPATH=/ucrt64/local/info:/ucrt64/share/info:/usr/local/info:/usr/share/info:/usr/info:/share/info
export PKG_CONFIG_PATH=/ucrt64/lib/pkgconfig:/ucrt64/share/pkgconfig
export PKG_CONFIG_SYSTEM_LIBRARY_PATH=/ucrt64/lib
export PKG_CONFIG_SYSTEM_INCLUDE_PATH=/ucrt64/include
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/libnvvp:/c/Windows/system32:/c/Windows:/c/Windows/System32/Wbem:/c/Windows/System32/WindowsPowerShell/v1.0/:/c/Windows/System32/OpenSSH/:/c/Program Files/Git/cmd:/c/Program Files/dotnet/:/d/tdmgcc/bin:/c/Program Files/eSpeak NG/:/c/Program Files (x86)/Windows Kits/10/Windows Performance Toolkit/:/c/Program Files/CMake/bin:/c/Program Files/NVIDIA Corporation/Nsight Compute 2024.1.0/:/c/Program Files (x86)/NVIDIA Corporation/PhysX/Common:/c/Users/g/AppData/Local/Microsoft/WindowsApps:/c/Users/g/.dotnet/tools:/D/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64:/ucrt64/bin:/usr/local/bin:/usr/bin:/bin"
make -j1 clean 2>/dev/null
CUDA_INC="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" \
CUDA_LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64" \
make -j1 cuda CC=/ucrt64/bin/gcc CUDA_ARCH=sm_89 2>&1 | tail -20
echo "BUILD_RC=$?"
