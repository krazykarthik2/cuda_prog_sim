@echo off
setlocal
rem ============================
rem  CONFIGURATION
rem ============================

rem Path to raylib extracted from raylib_win64_msvc.zip
set RAYLIB_ROOT="../raylib"

rem Target GPU architecture
set CUDA_ARCH=sm_86

rem CUDA toolkit paths (NO quotes here)
set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64
set CUDA_INC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include

rem MSVC Build Tools path (cl + vcvars64)
set MSVC_VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set MSVC_CL="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"

rem ============================
rem  COMPILE CUDA → OBJ
rem ============================

"%CUDA_BIN%\nvcc.exe" ^
  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" ^
  -c -Xcompiler "/MD" -o fluid_kernel.obj fluid_kernel.cu -arch=%CUDA_ARCH%

if errorlevel 1 goto :err

rem ============================
rem  INITIALIZE MSVC ENVIRONMENT
rem ============================
call %MSVC_VCVARS%
if errorlevel 1 goto :err

rem ============================
rem  COMPILE + LINK (raylib + CUDA)
rem ============================

%MSVC_CL% ^
  /EHsc /MD /O2 /DPLATFORM_WINDOWS ^
  /I"%RAYLIB_ROOT%\include" ^
  /I"%CUDA_INC%" ^
  hybrid.cpp fluid_kernel.obj ^
  /link ^
  /OUT:raylib_cuda.exe ^
  /LIBPATH:"%RAYLIB_ROOT%\lib" ^
  /LIBPATH:"%CUDA_LIB%" ^
  raylib.lib cudart_static.lib user32.lib gdi32.lib winmm.lib kernel32.lib

if errorlevel 1 goto :err

echo.
echo ===========================================
echo Build finished successfully!
echo Copy raylib.dll from %RAYLIB_ROOT%\bin to the EXE folder if needed.
echo ===========================================
echo.
del hybrid.obj
del fluid_kernel.obj
del raylib_cuda.exp
del raylib_cuda.lib
exit /b 0

:err
echo.
echo ***********  BUILD FAILED ***********
echo.
exit /b 1
