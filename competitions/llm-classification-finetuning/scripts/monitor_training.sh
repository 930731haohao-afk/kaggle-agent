#!/bin/bash
# Monitor training progress and alert on epoch completion

LOG_FILE="../training_log.txt"
CHECK_INTERVAL=60  # Check every 60 seconds

echo "🔍 Monitoring training progress..."
echo "Checking every ${CHECK_INTERVAL} seconds"
echo "Press Ctrl+C to stop monitoring"
echo ""

LAST_EPOCH_COUNT=0

while true; do
    # Check if process is still running
    if ! tmux has-session -t llm-training 2>/dev/null; then
        echo "⚠️  Training session ended!"
        break
    fi

    # Count completed epochs
    EPOCH_COUNT=$(grep -c "Epoch [0-9]*/[0-9]* - Train Loss" "$LOG_FILE" 2>/dev/null || echo "0")

    # Check for errors
    if grep -q "Error\|Exception\|Traceback" "$LOG_FILE" 2>/dev/null; then
        echo "❌ ERROR detected in training!"
        tail -30 "$LOG_FILE" | grep -A 10 "Error\|Exception\|Traceback"
        break
    fi

    # Check if new epoch completed
    if [ "$EPOCH_COUNT" -gt "$LAST_EPOCH_COUNT" ]; then
        echo ""
        echo "🎉 EPOCH $EPOCH_COUNT COMPLETED!"
        echo "----------------------------------------"
        grep "Epoch [0-9]*/[0-9]* - Train Loss" "$LOG_FILE" | tail -1
        echo ""

        # Show GPU status
        echo "GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
        echo ""

        LAST_EPOCH_COUNT=$EPOCH_COUNT
    else
        # Show progress indicator
        CURRENT_TIME=$(date +"%H:%M:%S")
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)
        echo "[$CURRENT_TIME] Training in progress... GPU: $GPU_UTIL | Epochs completed: $EPOCH_COUNT/2"
    fi

    sleep $CHECK_INTERVAL
done

echo ""
echo "Monitoring stopped."
