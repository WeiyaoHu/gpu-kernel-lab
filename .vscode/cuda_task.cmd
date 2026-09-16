@echo off
setlocal

set "MODE=%~1"
set "SOURCE=%~2"
set "PROJECT_ROOT=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "CUDA_DRIVE=Q:"

if "%SOURCE%"=="" (
    echo Usage: cuda_task.cmd build^|run source.cu
    exit /b 2
)

if not exist "%VSWHERE%" (
    echo Visual Studio Installer's vswhere.exe was not found.
    echo Install Visual Studio 2022 with the Desktop development with C++ workload.
    exit /b 3
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
if not defined VSINSTALL (
    echo A Visual Studio installation with the C++ toolchain was not found.
    exit /b 3
)
set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"

if exist %CUDA_DRIVE%\NUL (
    echo %CUDA_DRIVE% is already in use. Change CUDA_DRIVE in .vscode\cuda_task.cmd.
    exit /b 4
)

subst %CUDA_DRIVE% "%PROJECT_ROOT%" || exit /b 5
pushd %CUDA_DRIVE%\ || goto cleanup_error

if not exist build mkdir build
call "%VSDEVCMD%" -arch=x64 >nul || goto cleanup_error

set "NVCC=.cuda-env\Library\bin\nvcc.exe"
set "NVCC_COMPAT=-allow-unsupported-compiler"
if not exist "%NVCC%" (
    set "NVCC=nvcc.exe"
    set "NVCC_COMPAT="
    where nvcc.exe >nul 2>&1 || (
        echo nvcc.exe was not found. Install the CUDA Toolkit or create .cuda-env.
        goto cleanup_error
    )
)

"%NVCC%" --use-local-env %NVCC_COMPAT% -Xcompiler=/utf-8 -O3 "%SOURCE%" -o "build\%~n2.exe"
if errorlevel 1 goto cleanup_error

if /i "%MODE%"=="run" (
    echo.
    echo ===== Running %~n2 =====
    "build\%~n2.exe"
    if errorlevel 1 goto cleanup_error
)

popd
subst %CUDA_DRIVE% /d
exit /b 0

:cleanup_error
set "TASK_EXIT=%ERRORLEVEL%"
popd
subst %CUDA_DRIVE% /d
exit /b %TASK_EXIT%
