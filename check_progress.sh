#!/bin/bash

echo "=== OWLv2 Benchmark Progress ==="
echo ""
echo "Running processes:"
ps aux | grep "benchmark_owlv2.py" | grep -v grep | wc -l
echo ""

echo "Completed datasets:"
ls -1 predictions/owlv2/ | wc -l
echo ""

echo "Recent completions:"
ls -lt predictions/owlv2/ | head -10
echo ""

echo "GPU utilization:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo ""

echo "Latest activity from each GPU:"
for i in {0..7}; do
    echo "--- GPU $i ---"
    tail -3 gpu_$i.log 2>/dev/null | grep -E "(Processing|Saved|Error)" | tail -1
done
