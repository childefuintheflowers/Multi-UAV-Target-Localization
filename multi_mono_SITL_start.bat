@ECHO OFF

REM Set the path of the RflySim tools
SET PSP_PATH=D:\open-mmlab\mmdetection3d-master\multi-mono-uav

call activate open-mmlab

start %PSP_PATH%\python.exe drone1.py
start %PSP_PATH%\python.exe drone2.py
start %PSP_PATH%\python.exe drone3.py


exit
