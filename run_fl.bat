@echo off
REM Federated Learning Run Script for Gastric Cancer Classification (Windows)
REM This script starts the FL server and 3 hospital clients

echo Starting Federated Learning System for Gastric Cancer Classification
echo ==================================================================

REM Check if dataset is partitioned
if not exist "client\data\hospital_1" (
    echo Error: Dataset not partitioned yet!
    echo Please run the following command first:
    echo python partition_dataset.py
    pause
    exit /b 1
)

if not exist "client\data\hospital_2" (
    echo Error: Dataset not partitioned yet!
    echo Please run the following command first:
    echo python partition_dataset.py
    pause
    exit /b 1
)

if not exist "client\data\hospital_3" (
    echo Error: Dataset not partitioned yet!
    echo Please run the following command first:
    echo python partition_dataset.py
    pause
    exit /b 1
)

REM Main execution
if "%1"=="server" (
    echo Starting FL Server...
    python server\server.py --rounds 10 --min-clients 3 --local-epochs 2
) else if "%1"=="client" (
    if "%2"=="" (
        echo Usage: %0 client ^<hospital_id^>
        echo Hospital ID should be 1, 2, or 3
        pause
        exit /b 1
    )
    echo Starting Hospital %2 client...
    python client\client.py %2
) else if "%1"=="all" (
    echo Starting complete FL system...
    echo 1. Starting server in background...
    start "FL Server" python server\server.py --rounds 10 --min-clients 3 --local-epochs 2
    
    echo 2. Waiting 5 seconds for server to start...
    timeout /t 5 /nobreak > nul
    
    echo 3. Starting all clients...
    start "Hospital 1" python client\client.py 1
    start "Hospital 2" python client\client.py 2
    start "Hospital 3" python client\client.py 3
    
    echo 4. FL system is running!
    echo    - Check the opened windows for each component
    echo    - Close all windows to stop the system
    pause
) else if "%1"=="help" (
    echo Usage: %0 [server^|client^|all^|help]
    echo.
    echo Commands:
    echo   server     - Start only the FL server
    echo   client ^<id^> - Start a specific hospital client (1, 2, or 3)
    echo   all        - Start server and all clients (default)
    echo   help       - Show this help message
    echo.
    echo Examples:
    echo   %0                    # Start complete system
    echo   %0 server            # Start only server
    echo   %0 client 1          # Start hospital 1 client
) else (
    REM Default: start all
    echo Starting complete FL system...
    echo 1. Starting server in background...
    start "FL Server" python server\server.py --rounds 10 --min-clients 3 --local-epochs 2
    
    echo 2. Waiting 5 seconds for server to start...
    timeout /t 5 /nobreak > nul
    
    echo 3. Starting all clients...
    start "Hospital 1" python client\client.py 1
    start "Hospital 2" python client\client.py 2
    start "Hospital 3" python client\client.py 3
    
    echo 4. FL system is running!
    echo    - Check the opened windows for each component
    echo    - Close all windows to stop the system
    pause
)
