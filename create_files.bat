@echo off
:: Create files file1.txt to file10.txt
for /l %%i in (1,1,10) do (
    echo. > file%%i.txt
    echo Created file%%i.txt
)

pause