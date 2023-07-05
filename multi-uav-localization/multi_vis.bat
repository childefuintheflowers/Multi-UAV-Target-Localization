@ECHO OFF

REM Set the path of the RflySim tools
SET PSP_PATH=D:\PX4PSP
D:

REM kill all applications when press a key
tasklist|find /i "CopterSim.exe" && taskkill /im "CopterSim.exe"
tasklist|find /i "QGroundControl.exe" && taskkill /f /im "QGroundControl.exe"
tasklist|find /i "RflySim3D.exe" && taskkill /f /im "RflySim3D.exe"

REM UE4Path
start %PSP_PATH%\RflySim3D\RflySim3D.exe

pause
tasklist|find /i "RflySim3D.exe" && taskkill /f /im "RflySim3D.exe"
