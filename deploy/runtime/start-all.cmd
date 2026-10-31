@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  Rice_system - 一键启动：数据库 -> 后端 -> 前端
rem ============================================================
set "DEPLOY=%~dp0"

echo === 1/3 检查本机 PostgreSQL 5432 ===
call "%DEPLOY%\start-db.cmd"

echo === 2/3 启动后端 8000 ===
start "rice-backend" cmd /k "%DEPLOY%\start-backend.cmd"

echo       等待后端就绪...
set /a tries=0
:wait_backend
set /a tries+=1
timeout /t 2 /nobreak >nul
curl -s -o nul http://127.0.0.1:8000/ && goto backend_ok
if %tries% lss 30 goto wait_backend
echo [警告] 后端 60 秒内仍未就绪，请查看 rice-backend 窗口
goto start_web

:backend_ok
echo       后端已就绪

:start_web
echo === 3/3 启动前端 5173 ===
start "rice-frontend" cmd /k "%DEPLOY%\start-frontend.cmd"

timeout /t 3 /nobreak >nul
echo.
echo 访问地址: http://localhost:5173/    接口文档: http://localhost:8000/docs
call "%DEPLOY%\status.cmd"
endlocal
