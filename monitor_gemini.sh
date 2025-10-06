#!/bin/bash
# Monitor Gemini benchmark progress

echo "========================================"
echo "GEMINI 2.5 PRO BENCHMARK MONITOR"
echo "========================================"
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "benchmark_gemini.py" > /dev/null; then
    echo "✓ Benchmark is RUNNING"
    PID=$(pgrep -f "benchmark_gemini.py")
    echo "  PID: $PID"
else
    echo "✗ Benchmark is NOT running"
fi

echo ""
echo "========================================"
echo "PREDICTIONS PROGRESS"
echo "========================================"

# Count completed datasets
if [ -d "predictions/gemini" ]; then
    COMPLETED=$(find predictions/gemini -name "predictions.json" | wc -l)
    echo "Completed datasets: $COMPLETED / 100"

    # Calculate percentage
    PERCENT=$((COMPLETED * 100 / 100))
    echo "Progress: $PERCENT%"

    # Calculate estimated cost so far
    COST_PER_DATASET=0.44
    ESTIMATED_COST=$(echo "$COMPLETED * $COST_PER_DATASET" | bc)
    echo "Estimated cost so far: \$${ESTIMATED_COST}"

    echo ""
    echo "Recent completions:"
    find predictions/gemini -name "predictions.json" -printf "%T@ %p\n" | sort -n | tail -5 | cut -d'/' -f3
else
    echo "No predictions directory found yet"
fi

echo ""
echo "========================================"
echo "RECENT LOG OUTPUT"
echo "========================================"
if [ -f "gemini_benchmark.log" ]; then
    tail -20 gemini_benchmark.log
else
    echo "No log file found"
fi

echo ""
echo "========================================"
echo "To monitor live: tail -f gemini_benchmark.log"
echo "To stop: pkill -f benchmark_gemini.py"
echo "========================================"
