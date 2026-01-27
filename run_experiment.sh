#!/bin/bash
#OAR -n NEVOptExp1
#OAR -l nodes=10,walltime=12:00:00
#OAR -t exotic
#OAR -p neowise
#OAR -t night
#OAR --stdout ./logs/%jobid%.stdout
#OAR --stderr ./logs/%jobid%.stderr

# OAR job script to run the benchmark experiment using NEVO algorithm
# -l nodes=4,walltime=12:00:00

# Experiment parameters (define once, use everywhere)
PROBLEMS="1-24"
INSTANCES="1-15"
DIMENSIONS="2,3,5,10,20,40"
RUNS=1
TIME=20.0
CORES=0
ALGORITHM_NAME="NEVO"
OUTPUT_DIR="benchmark_results_cocoex"

# Activate the virtual environment
echo "Activating the virtual environment and setting up the environment..."
source setup.sh
echo "Environment setup complete."

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
    --algorithm-name $ALGORITHM_NAME \
    --output-dir $OUTPUT_DIR

echo "Experiment completed."
echo "Benchmark experiment finished."

# End of script