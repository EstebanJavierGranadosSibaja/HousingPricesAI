@echo off
REM Mostrar el estado actual
git status

REM Solicitar al usuario el hash del commit
set /p commitHash=Introduce el hash del commit que deseas usar para el reset: 

REM Hacer un hard reset al commit especificado
git reset --soft %commitHash%

REM Validar que el hash no este vacio
if "%commitHash%"=="" (
    echo Error: No se proporciono un hash de commit.
    pause
    exit /b
)

echo Los registros despues del commit han sido borrados, pero los cambios estan en tu directorio de trabajo.
pause
