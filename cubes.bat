@echo off
setlocal
cd /d "%~dp0"
set CMD=%1
if "%CMD%"=="" goto ayuda
shift
set ARGS=
:juntar
if "%~1"=="" goto listo
set ARGS=%ARGS% %1
shift
goto juntar
:listo

set CP=build\classes
if exist target\classes\pe set CP=target\classes

if /i "%CMD%"=="compilar" goto compilar
if /i "%CMD%"=="nodo-python" ( python nodo_python\nodo.py %ARGS% & goto fin )
if /i "%CMD%"=="nodo-js" ( node nodo_js\nodo.js %ARGS% & goto fin )
if /i "%CMD%"=="nodo-go" ( cd nodo_go & go build -o nodo_go.exe . & nodo_go.exe %ARGS% & goto fin )
if not exist %CP%\pe call :compilar_sub
if /i "%CMD%"=="servidor" ( java -cp %CP% pe.uni.cubes.servidor.ServidorCentral %ARGS% & goto fin )
if /i "%CMD%"=="nodo-java" ( java -cp %CP% pe.uni.cubes.nodo.NodoJava %ARGS% & goto fin )
if /i "%CMD%"=="cliente" ( java -cp %CP% pe.uni.cubes.cliente.ClienteChat %ARGS% & goto fin )
if /i "%CMD%"=="estres" ( java -cp %CP% pe.uni.cubes.herramientas.PruebaEstres %ARGS% & goto fin )
if /i "%CMD%"=="graficas" ( java -cp %CP% pe.uni.cubes.herramientas.Graficas %ARGS% & goto fin )
if /i "%CMD%"=="sinteticos" ( java -cp %CP% pe.uni.cubes.herramientas.GenerarSinteticos %ARGS% & goto fin )
goto ayuda

:compilar
call :compilar_sub
goto fin

:compilar_sub
if not exist build\classes mkdir build\classes
dir /s /b src\main\java\*.java > build\fuentes.txt
javac --release 8 -Xlint:-options -encoding UTF-8 -d build\classes @build\fuentes.txt 2>nul || javac -encoding UTF-8 -d build\classes @build\fuentes.txt
set CP=build\classes
echo Compilado en build\classes
exit /b 0

:ayuda
echo Uso: cubes.bat ^<compilar^|servidor^|nodo-java^|nodo-python^|nodo-js^|nodo-go^|cliente^|estres^|graficas^|sinteticos^> [argumentos]
:fin
endlocal
