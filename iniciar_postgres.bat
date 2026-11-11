@echo off
set PGBIN=C:\Program Files\PostgreSQL\18\bin
set PGDATA=C:\Program Files\PostgreSQL\18\data
"%PGBIN%\pg_ctl.exe" status -D "%PGDATA%" >nul 2>&1
if %errorlevel%==0 (
  echo PostgreSQL ya esta en ejecucion.
) else (
  if exist "%PGDATA%\postmaster.pid" del "%PGDATA%\postmaster.pid"
  start "" /b "%PGBIN%\pg_ctl.exe" start -D "%PGDATA%" -l "%PGDATA%\log\pg_ctl.log"
  echo PostgreSQL iniciado.
)
