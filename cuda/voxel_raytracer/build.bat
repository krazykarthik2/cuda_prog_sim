@echo off
setlocal

rem Path to raylib
set RAYLIB_ROOT=../raylib

rem Environment Setup
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    echo Error: vcvars64.bat not found at %VCVARS%
    goto :err
)

call %VCVARS%

rem Compile + Link
echo.
echo [Compiling and linking Voxel Ray Tracer...]
cl.exe /O2 /W3 /std:c++14 ^
  /I "%RAYLIB_ROOT%\include" ^
  main.cpp ^
  /Fe:voxel_raytracer.exe ^
  /link ^
  /LIBPATH:"%RAYLIB_ROOT%\lib" ^
  raylibdll.lib opengl32.lib gdi32.lib winmm.lib user32.lib shell32.lib

if errorlevel 1 goto :err

copy /y "%RAYLIB_ROOT%\lib\raylib.dll" .

echo.
echo Build finished successfully!
exit /b 0

:err
echo.
echo ***********  BUILD FAILED ***********
exit /b 1
