@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  Rice_system - 项目不管理本机 PostgreSQL 服务
rem ============================================================
echo [跳过] PostgreSQL 由 Windows 服务管理，请使用“服务”控制台维护 5432 实例。
endlocal
