@echo off
setlocal
rem ============================
rem  CONFIGURATION
rem ============================

rem Path to raylib
set RAYLIB_ROOT=../raylib

rem Target GPU architecture
set CUDA_ARCH=compute_75

rem CUDA toolkit paths
set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\lib\x64
set CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\include

rem ============================
rem  ENVIRONMENT SETUP (MSVC)
rem ============================

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    echo Error: vcvars64.bat not found at %VCVARS%
    goto :err
)

call %VCVARS%

rem ============================
rem  COMPILE + LINK (MSVC)
rem ============================

echo.
echo [Compiling and linking with MSVC...]
cl.exe /O2 /W3 /std:c++14 ^
  /I "%RAYLIB_ROOT%\include" ^
  main.cpp ^
  /Fe:dithering_playground.exe ^
  /link ^
  /LIBPATH:"%RAYLIB_ROOT%\lib" ^
  raylibdll.lib opengl32.lib gdi32.lib winmm.lib user32.lib shell32.lib

if errorlevel 1 goto :err

copy /y "%RAYLIB_ROOT%\lib\raylib.dll" .

echo.
echo ===========================================
echo Build finished successfully!
echo ===========================================
echo.
del dither_kernel.obj 2>nul
exit /b 0

:err
echo.
echo ***********  BUILD FAILED ***********
echo.
exit /b 1
