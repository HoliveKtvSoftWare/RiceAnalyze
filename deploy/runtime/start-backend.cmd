@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  Rice_system - 启动 FastAPI 后端 0.0.0.0:8000
rem  从脚本位置定位后端，允许从任意工作目录启动。
rem  优先 RICE_PYTHON，其次项目 .venv，最后当前环境的 python。
rem  不要加 --reload，受限环境下 reload 子进程会 EPERM。
rem ============================================================
for %%I in ("%~dp0..\..") do set "APP_DIR=%%~fI"
cd /d "%APP_DIR%"
set "PY=%RICE_PYTHON%"
if not defined PY if exist "%APP_DIR%\.venv\Scripts\python.exe" set "PY=%APP_DIR%\.venv\Scripts\python.exe"
if not defined PY set "PY=python"
set "YOLO_CONFIG_DIR=%APP_DIR%\.ultralytics"

if exist "%PY%" goto python_ok
where "%PY%" >nul 2>&1
if errorlevel 1 (
  echo [错误] 找不到 Python 解释器: %PY%
  echo        请先激活后端Python环境，或设置RICE_PYTHON。
  exit /b 1
)
:python_ok
if not exist "%APP_DIR%\models_yolo\stem\yolov8s-1.pt" echo [提示] 默认茎秆权重不存在，请检查 .env 中的 YOLO_MODEL_PATH
if not exist "%APP_DIR%\models_yolo\leaf\leaf-our-best.pt" echo [提示] 默认剑叶 OUR 权重不存在，请检查 .env 中的 LEAF_MODEL_PATH

echo [启动] 后端 uvicorn app.main:app -^> http://0.0.0.0:8000    接口文档 http://127.0.0.1:8000/docs
call "%PY%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips=127.0.0.1
set "BACKEND_EXIT=%ERRORLEVEL%"
endlocal & exit /b %BACKEND_EXIT%
