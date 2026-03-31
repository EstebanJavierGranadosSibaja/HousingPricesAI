@echo off
REM Obtener la ruta del directorio donde se encuentra este archivo .bat
SET "CURRENT_DIR=%~dp0"

REM Verificar si Python está instalado
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python no está instalado o no está configurado en el PATH.
    echo Por favor instala Python antes de continuar.
    pause
    exit /b
)

REM Crear un entorno virtual en la carpeta 'venv'
echo Creando el entorno virtual en la carpeta 'venv'...
python -m venv "%CURRENT_DIR%venv"

REM Activar el entorno virtual
echo Activando el entorno virtual...
call "%CURRENT_DIR%venv\Scripts\activate"

REM Actualizar pip, setuptools y wheel
echo Actualizando pip, setuptools y wheel...
pip install --upgrade pip setuptools wheel

REM Instalar dependencias si existe requirements.txt
if exist "%CURRENT_DIR%requirements.txt" (
    echo Instalando dependencias desde requirements.txt...
    pip install -r "%CURRENT_DIR%requirements.txt"
) else (
    echo No se encontró requirements.txt. Entorno creado sin dependencias.
)

REM Ejecutar Odoo
echo Preparando para ejecutar Odoo...
echo Asegúrese de haber configurado PostgreSQL y las rutas de los complementos personalizados.
echo Ejecutando Odoo...

REM Cambiar al directorio donde está el archivo odoo-bin
cd /d "%CURRENT_DIR%"

REM Reemplaza dbuser, dbpassword y mydb con tus propios valores
python odoo-bin -r admin-w admin --addons-path=addons -d LogiDB

REM Finalización
echo El entorno virtual ha sido creado, las dependencias han sido instaladas y Odoo está ejecutándose.
pause
