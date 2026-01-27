#!/bin/bash
#OAR -n NEVOptExp1
#OAR -l nodes=1,walltime=1:00:00
#OAR -t exotic
#OAR -p neowise
#OAR --stdout ./logs/%jobid%.stdout
#OAR --stderr ./logs/%jobid%.stderr

# OAR job script to run the benchmark experiment using NEVO algorithm
# -l nodes=4,walltime=12:00:00

# Activate the virtual environment
echo "Activating the virtual environment and setting up the environment..."
source setup.sh
echo "Environment setup complete."

# Starting the experiment with specified parameters
echo "Starting the benchmark experiment..."
python ./nevo/examples/benchmark_experiment.py \
    --suite cocoex \
    --problems 1-24 \
    --instances 1-15 \
    --dimensions 2,3,5,10,20,40 \
    --time 20.0 \
    --runs 1 \
    --cores 0 \
    --algorithm-name NEVO \
    --output-dir benchmark_results_cocoex

echo "Experiment completed."
echo "Benchmark experiment finished."

# End of script