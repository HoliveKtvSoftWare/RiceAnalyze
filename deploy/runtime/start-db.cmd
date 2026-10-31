@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  Rice_system - 检查本机 PostgreSQL 服务 127.0.0.1:5432
rem  数据库由 Windows PostgreSQL 服务管理，项目不再启动独立数据目录。
rem ============================================================
cd /d "%~dp0..\.."
if not defined PGPORT set "PGPORT=5432"
if defined PGBIN goto use_pgbin
where pg_isready.exe >nul 2>&1
if not errorlevel 1 goto pg_ready
for /f "delims=" %%D in ('dir /b /ad /o-n "%ProgramFiles%\PostgreSQL" 2^>nul') do if not defined PGBIN if exist "%ProgramFiles%\PostgreSQL\%%D\bin\pg_isready.exe" set "PGBIN=%ProgramFiles%\PostgreSQL\%%D\bin"
if not defined PGBIN (
  echo [错误] 找不到pg_isready，请将PostgreSQL的bin目录加入PATH或设置PGBIN。
  exit /b 1
)
:use_pgbin
set "PATH=%PGBIN%;%PATH%"
:pg_ready

pg_isready.exe -h 127.0.0.1 -p %PGPORT% >nul 2>&1
if errorlevel 1 (
  echo [错误] 本机 PostgreSQL 未就绪: 127.0.0.1:%PGPORT%
  exit /b 1
)
echo [正常] 本机 PostgreSQL 已就绪: 127.0.0.1:%PGPORT%
endlocal
