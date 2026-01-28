#!/bin/bash
#OAR -n NEVOptExp1
#OAR -l nodes=9,walltime=12:00:00
#OAR -p nova
#OAR -t exotic
#OAR --stdout ./logs/%jobid%.stdout
#OAR --stderr ./logs/%jobid%.stderr

# OAR job script to run the benchmark experiment using NEVO algorithm
# -t night
# -l nodes=4,walltime=12:00:00

# Experiment parameters (define once, use everywhere)
PROBLEMS="1-24"
INSTANCES="1-15"
DIMENSIONS="2,3,5,10,20,40"
RUNS=1
#TIME=10.0
TIME=10.0
CORES=0
ALGORITHM_NAME="NEVO"
#OUTPUT_DIR="benchmark_results_cocoex-reduced-time"
OUTPUT_DIR="benchmark_results_cocoex"

# Activate the virtual environment
echo "Activating the virtual environment and setting up the environment..."
source setup.sh
echo "Environment setup complete."

# Check if accelerated mode is available and install if needed
echo "Checking/installing nengo-dl and tensorflow..."
uv pip install nengo-dl tensorflow --quiet || {
    echo "Warning: Failed to install nengo-dl/tensorflow. Continuing without GPU acceleration."
}

# Verify nengo-dl is available
if python -c "import nengo_dl" 2>/dev/null; then
    echo "nengo-dl is available. Using accelerated mode."
    USE_DL="--use-dl"
    # Note: When using GPU acceleration, use fewer cores as GPU handles parallelism
    # Uncomment below to force single core with GPU:
    CORES=1
else
    echo "nengo-dl is not available. Running without GPU acceleration."
    USE_DL=""
fi

#USE_DL=""

# Starting the experiment with specified parameters
echo "Checking progress of previous runs..."
python ./check_progress.py \
    --problems $PROBLEMS \
    --instances $INSTANCES \
    --dimensions $DIMENSIONS \
    --runs $RUNS \
    --output-dir $OUTPUT_DIR

echo "Starting the benchmark experiment..."
python ./nevo/examples/benchmark_experiment.py \
    --suite cocoex \
    --problems $PROBLEMS \
    --instances $INSTANCES \
    --dimensions $DIMENSIONS \
    --time $TIME \
    --runs $RUNS \
    --cores $CORES \
    $USE_DL \
    --algorithm-name $ALGORITHM_NAME \
    --output-dir $OUTPUT_DIR

echo "Experiment completed."
echo "Benchmark experiment finished."

# End of script