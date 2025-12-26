@echo off
echo STARTING DEBUG RUN > debug_output.txt
echo DATE: %DATE% %TIME% >> debug_output.txt
echo. >> debug_output.txt

echo 1. Checking Python version >> debug_output.txt
python --version >> debug_output.txt 2>&1

echo. >> debug_output.txt
echo 2. Running sign.py >> debug_output.txt
python sign.py >> debug_output.txt 2>&1

echo. >> debug_output.txt
echo DONE >> debug_output.txt
pause
