@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
rem ============================================================
rem  Rice_system - 部署自检
rem  说明：psql 的 SQL 里不要写中文，cmd 会把命令行按 ANSI 代码页
rem        传给 psql，导致 UTF8 客户端编码报“无效的字节顺序”；
rem        中文标签一律用 echo 输出。
rem
rem  退出码：0 = 全部正常；1 = 有检查项失败。
rem  注意：反代那项**未登录本就该是 401**，属于正常结果，不再计入失败，
rem        也不会再让脚本带着 curl 的 4xx 退出码结束（此前会误报失败）。
rem ============================================================
set "FAILS=0"
set "HOST=127.0.0.1"

echo --- 端口监听 ---
for %%P in (5432 8000 5173) do (
  netstat -ano -p tcp | findstr /c:":%%P " | findstr /c:"LISTENING" >nul
  if errorlevel 1 (
    echo   [未启动] 端口 %%P
    set /a FAILS+=1
  ) else (
    echo   [正常] 端口 %%P
  )
)

echo --- 服务响应 ---
rem 每个检查单独判定：先看 curl 自身退出码（连通性），再看 HTTP 状态码
set "CODE="
curl -s -o nul -w "%%{http_code}" -o nul http://%HOST%:8000/ > "%TEMP%\rice_status_code.txt" 2>nul
set /p CODE=<"%TEMP%\rice_status_code.txt"
if "%CODE%"=="200" (echo   后端      http://%HOST%:8000/                     -^> 200) else (echo   后端      http://%HOST%:8000/                     -^> %CODE%  应为 200& set /a FAILS+=1)

set "CODE="
curl -s -o nul -w "%%{http_code}" -o nul http://%HOST%:5173/ > "%TEMP%\rice_status_code.txt" 2>nul
set /p CODE=<"%TEMP%\rice_status_code.txt"
if "%CODE%"=="200" (echo   前端      http://%HOST%:5173/                     -^> 200) else (echo   前端      http://%HOST%:5173/                     -^> %CODE%  应为 200& set /a FAILS+=1)

set "CODE="
curl -s -o nul -w "%%{http_code}" -o nul http://%HOST%:5173/api/analysis/history > "%TEMP%\rice_status_code.txt" 2>nul
set /p CODE=<"%TEMP%\rice_status_code.txt"
if "%CODE%"=="401" (
  echo   API 反代  http://%HOST%:5173/api/analysis/history -^> 401  正常（未登录应为 401）
) else (
  echo   API 反代  http://%HOST%:5173/api/analysis/history -^> %CODE%  应为 401（反代不通或后端异常）
  set /a FAILS+=1
)

echo --- 数据库 ---
call "%~dp0start-db.cmd" >nul 2>&1
if errorlevel 1 (
  echo   [未启动] 本机 PostgreSQL %HOST%:5432 连接失败
  set /a FAILS+=1
) else (
  echo   [正常] 本机 PostgreSQL %HOST%:5432
)

echo.
if !FAILS! EQU 0 (
  echo 自检结果: 全部正常
) else (
  echo 自检结果: !FAILS! 项异常，详见上面标记
)
echo.
echo 离线推理测试工具已移出后端仓库，需要时按外部归档的使用说明恢复。

if !FAILS! EQU 0 (endlocal & exit /b 0) else (endlocal & exit /b 1)
