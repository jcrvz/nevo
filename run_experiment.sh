#!/bin/bash
#OAR -n NEVOpt-Benchmark
#OAR -l nodes=9,walltime=10:00:00
#OAR -p neowise
#OAR -t exotic
#OAR -t night
#OAR --stdout ./logs/%jobid%.stdout
#OAR --stderr ./logs/%jobid%.stderr

# OAR job script to run the benchmark experiment using NEVO algorithm
# -t night
# -t exotic
# -l nodes=4,walltime=12:00:00

# Default experiment parameters
PROBLEMS="1-24"
INSTANCES="1-15"
DIMENSIONS="2,3,5,10,20,40"
RUNS=1
TIME=20.0  # Increased for better convergence (gives ~1M evals with pop_size=100)
CORES=0
ALGORITHM_NAME="NEVO"
# COCO suite: bbob | bbob-noisy | bbob-largescale | bbob-mixint | bbob-constrained | bbob-biobj | bbob-biobj-ext | sbox-cost
COCO_SUITE="bbob"

# --- Variant parameters (defaults match original trad implementation) ---
# Operator mode: trad | nm_dual | nm_softmix
OPERATOR_MODE="trad"
# TD learning: true | false
TD_ENABLED=true
# TD(λ) coefficient: 0.0 = TD(0), e.g. 0.9 = TD(λ)
TD_LAMBDA=0.0

# --- Load config file if provided as first argument ---
# Usage: bash run_experiment.sh configs/nm_dual_td0.conf
# Available configs in configs/: trad_eps_greedy, trad_td0, trad_td_lambda,
#   nm_dual_eps_greedy, nm_dual_td0, nm_dual_td_lambda,
#   nm_softmix_eps_greedy, nm_softmix_td0, nm_softmix_td_lambda
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Loading config: $1"
    source "$1"
elif [ -n "$1" ]; then
    echo "Error: config file '$1' not found."
    exit 1
fi

# Derived parameters — computed after config is loaded so overrides take effect
# Use the full ALGORITHM_NAME (minus the "NEVO_" prefix) so each variant gets
# its own folder, e.g. benchmark_results_bbob_nm_softmix_eps_greedy
VARIANT="${ALGORITHM_NAME#NEVO_}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results_${COCO_SUITE//-/_}_${VARIANT}}"

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

# I'm mainly using CPU for now ...
USE_DL=""
CORES=0

# Starting the experiment with specified parameters
echo "Checking progress of previous runs..."
python ./check_progress.py \
    --problems $PROBLEMS \
    --instances $INSTANCES \
    --dimensions $DIMENSIONS \
    --runs $RUNS \
    --output-dir $OUTPUT_DIR

echo "Starting the benchmark experiment..."
# Build optional flags
TD_FLAG=""
if [ "$TD_ENABLED" = false ]; then
    TD_FLAG="--no-td"
fi

python ./examples/benchmark_experiment.py \
    --suite cocoex \
    --coco-suite $COCO_SUITE \
    --problems $PROBLEMS \
    --instances $INSTANCES \
    --dimensions $DIMENSIONS \
    --time $TIME \
    --runs $RUNS \
    --cores $CORES \
    $USE_DL \
    --operator-mode $OPERATOR_MODE \
    --td-lambda $TD_LAMBDA \
    $TD_FLAG \
    --algorithm-name $ALGORITHM_NAME \
    --output-dir $OUTPUT_DIR

echo "Experiment completed."
echo "Benchmark experiment finished."

# End of script