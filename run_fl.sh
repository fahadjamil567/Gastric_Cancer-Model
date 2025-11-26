#!/bin/bash

# Federated Learning Run Script for Gastric Cancer Classification
# This script starts the FL server and 3 hospital clients

echo "Starting Federated Learning System for Gastric Cancer Classification"
echo "=================================================================="

# Check if dataset is partitioned
if [ ! -d "client/data/hospital_1" ] || [ ! -d "client/data/hospital_2" ] || [ ! -d "client/data/hospital_3" ]; then
    echo "Error: Dataset not partitioned yet!"
    echo "Please run the following command first:"
    echo "python partition_dataset.py"
    exit 1
fi

# Function to start server
start_server() {
    echo "Starting FL Server..."
    python server/server.py --rounds 10 --min-clients 3 --local-epochs 2
}

# Function to start client
start_client() {
    local hospital_id=$1
    echo "Starting Hospital $hospital_id client..."
    python client/client.py $hospital_id
}

# Function to start all clients in background
start_all_clients() {
    echo "Starting all hospital clients..."
    start_client 1 &
    start_client 2 &
    start_client 3 &
}

# Main execution
case "${1:-all}" in
    "server")
        start_server
        ;;
    "client")
        if [ -z "$2" ]; then
            echo "Usage: $0 client <hospital_id>"
            echo "Hospital ID should be 1, 2, or 3"
            exit 1
        fi
        start_client $2
        ;;
    "all")
        echo "Starting complete FL system..."
        echo "1. Starting server in background..."
        start_server &
        SERVER_PID=$!
        
        echo "2. Waiting 5 seconds for server to start..."
        sleep 5
        
        echo "3. Starting all clients..."
        start_all_clients
        
        echo "4. FL system is running!"
        echo "   - Server PID: $SERVER_PID"
        echo "   - Press Ctrl+C to stop all processes"
        
        # Wait for user interrupt
        trap "echo 'Stopping FL system...'; kill $SERVER_PID; killall python; exit 0" INT
        wait
        ;;
    "help")
        echo "Usage: $0 [server|client|all|help]"
        echo ""
        echo "Commands:"
        echo "  server     - Start only the FL server"
        echo "  client <id>- Start a specific hospital client (1, 2, or 3)"
        echo "  all        - Start server and all clients (default)"
        echo "  help       - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Start complete system"
        echo "  $0 server            # Start only server"
        echo "  $0 client 1          # Start hospital 1 client"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
