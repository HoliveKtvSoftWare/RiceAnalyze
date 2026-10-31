@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  Rice_system - 启动前端服务器 0.0.0.0:5173
rem  托管 RiceAnylazeWeb\dist，并把 /api、/static 反向代理到后端。
rem ============================================================
set "DEPLOY_DIR=%~dp0"
set "WEB_ROOT=%~dp0..\..\..\RiceAnylazeWeb\dist"
set "PORT=5173"
set "BACKEND_ORIGIN=http://127.0.0.1:8000"

if not exist "%WEB_ROOT%\index.html" (
  echo [错误] 前端产物不存在: %WEB_ROOT%
  echo        请先构建: cd /d %~dp0..\..\..\RiceAnylazeWeb ^&^& npm run build
  exit /b 1
)

echo [启动] 前端 http://0.0.0.0:%PORT%/    静态目录 %WEB_ROOT%
node "%DEPLOY_DIR%\serve.mjs" "%WEB_ROOT%"
endlocal
