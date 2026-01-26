# NEVO Server Experiment Guide

## Pre-flight Checklist

Before running a large experiment on the server, verify the following:

### 1. Dependencies
```bash
# Install all dependencies
pip install -e ".[dev]"

# Or using requirements
pip install -r requirements.txt
pip install cocoex cocopp pandas
```

### 2. Test the Setup
```bash
# Run unit tests
python -m pytest tests/ -v

# Quick benchmark test (should complete in ~2 minutes)
python ./nevo/examples/benchmark_experiment.py \
    --suite cocoex \
    --problems 1 \
    --instances 1 \
    --dimensions 2 \
    --time 1.0 \
    --runs 1 \
    --cores 1
```

### 3. Test Parallelisation (Important!)
```bash
# Test with 2 cores to verify parallel execution works
python ./nevo/examples/benchmark_experiment.py \
    --suite cocoex \
    --problems 1-2 \
    --instances 1-2 \
    --dimensions 2 \
    --time 1.0 \
    --runs 1 \
    --cores 2
```

**Expected output should show:**
- `[Batch 1/2]` and `[Batch 2/2]` running in parallel
- `Merged X batch folders into exdata/NEVO` at the end
- No "cache lock" or "semaphore leak" warnings

---

## Parallelisation Notes

The benchmark uses `multiprocessing` with:
- **Spawn context** (avoids fork issues with Nengo on macOS/Linux)
- **Disabled Nengo cache** (avoids lock conflicts between workers)
- **COCO BatchScheduler** (ensures proper problem distribution)

Each core runs a separate batch that handles a subset of problems.
After completion, batch folders are automatically merged for cocopp postprocessing.

---

## Running Large Experiments

### COCO Benchmark (Recommended for Publications)

#### Full BBOB Suite (24 functions × 15 instances × 6 dimensions)
```bash
python ./nevo/examples/benchmark_experiment.py \
    --suite cocoex \
    --problems 1-24 \
    --instances 1-15 \
    --dimensions 2,3,5,10,20,40 \
    --time 20.0 \
    --runs 5 \
    --cores 0 \
    --algorithm-name NEVO
```

#### Subset for Testing (faster)
```bash
python ./nevo/examples/benchmark_experiment.py \
    --suite cocoex \
    --problems 1-6 \
    --instances 1-5 \
    --dimensions 2,5,10 \
    --time 10.0 \
    --runs 3 \
    --cores 4 \
    --algorithm-name NEVO
```

### IOH Benchmark
```bash
python ./nevo/examples/benchmark_experiment.py \
    --suite ioh \
    --problems 1-24 \
    --instances 1-5 \
    --dimensions 10,20 \
    --time 20.0 \
    --runs 5 \
    --cores 0
```

---

## CLI Arguments Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--suite` | Benchmark suite: `ioh` or `cocoex` | `ioh` |
| `--problems` | Problem IDs (supports ranges: `1-24`, `1,2,5`) | `1` |
| `--instances` | Problem instances (supports ranges) | `1-5` |
| `--dimensions` | Problem dimensions (COCO: 2,3,5,10,20,40) | `2` |
| `--time` | Simulation time in seconds | `20.0` |
| `--runs` | Independent runs per problem | `1` |
| `--cores` | CPU cores (0 = all available) | `1` |
| `--output-dir` | Output directory | `benchmark_results_<suite>` |
| `--algorithm-name` | Algorithm name for COCO logging | `NEVO` |

---

## Output Structure

### COCO Suite
```
benchmark_results_cocoex/
├── all_results.csv           # Combined results
├── summary_statistics.csv    # Aggregated statistics
├── results_f01_i01_2D_run01.csv  # Individual results
└── ...

exdata/
└── NEVO/                     # COCO data for cocopp
    ├── bbobexp_f1.info
    ├── data_f1/
    └── ...
```

### IOH Suite
```
benchmark_results_ioh/
├── all_results.csv
├── summary_statistics.csv
├── results_f01_i01_10D.csv
├── benchmark_ioh_f01_i01_10D.png  # Plots (first run only)
└── ...
```

---

## Postprocessing

### COCO Results (cocopp)
```bash
# Basic postprocessing
python -m cocopp exdata/NEVO

# Compare with reference algorithms
python -m cocopp exdata/NEVO bbob/2009/RANDOMSEARCH bbob/2009/PSO!

# Results appear in ppdata/ folder
```

### Analysis
```python
import pandas as pd

# Load results
df = pd.read_csv('benchmark_results_cocoex/all_results.csv')

# Summary by problem
summary = df.groupby(['problem_id', 'dimension']).agg({
    'best_fitness': ['mean', 'std', 'min'],
    'total_evaluations': 'mean',
    'wall_time': 'mean'
})
print(summary)
```

---

## Server-Specific Tips

### Running in Background (nohup)
```bash
nohup python ./nevo/examples/benchmark_experiment.py \
    --suite cocoex \
    --problems 1-24 \
    --instances 1-15 \
    --dimensions 2,3,5,10,20,40 \
    --time 20.0 \
    --runs 5 \
    --cores 0 \
    --algorithm-name NEVO \
    > experiment.log 2>&1 &
```

### Using Screen/Tmux
```bash
# Create session
screen -S nevo_experiment

# Run experiment
python ./nevo/examples/benchmark_experiment.py ...

# Detach: Ctrl+A, D
# Reattach: screen -r nevo_experiment
```

### Monitoring Progress
```bash
# Watch log file
tail -f experiment.log

# Count completed experiments
ls benchmark_results_cocoex/results_*.csv | wc -l
```

---

## Current Operator Configuration

The optimiser uses **13 operators** by default:

### Exploration Operators (7)
| Short | Name | Complexity |
|-------|------|------------|
| RS | RandomSearch | 1 |
| LF | LevyFlight | 3 |
| GX | GeneticCrossover | 4 |
| DE | DifferentialEvolution | 5 |
| CF | CentralForce | 6 |
| FA | FireflyAlgorithm | 7 |
| GS | GravitationalSearch | 8 |

### Exploitation Operators (6)
| Short | Name | Complexity |
|-------|------|------------|
| LW | LocalRandomWalk | 2 |
| GM | GeneticMutation | 3 |
| SA | SimulatedAnnealing | 4 |
| TS | TabuSearch | 5 |
| PS | ParticleSwarm | 6 |
| SO | SpiralOptimisation | 7 |

---

## Estimated Runtime

| Configuration | Problems | Instances | Dims | Runs | Time/exp | Est. Total |
|---------------|----------|-----------|------|------|----------|------------|
| Quick test | 1 | 1 | 1 | 1 | 1s | ~15s |
| Small | 6 | 5 | 3 | 3 | 10s | ~4.5h |
| Medium | 24 | 5 | 3 | 5 | 20s | ~25h |
| Full BBOB | 24 | 15 | 6 | 5 | 20s | ~150h |

*Times are approximate and depend on hardware. Using multiple cores significantly reduces wall time.*

---

## Troubleshooting

### Memory Issues
- Reduce `--cores` to limit parallel processes
- Reduce `--dimensions` (higher dims use more memory)

### COCO Observer Errors
- Ensure `exdata/` directory is writable
- Check disk space for COCO data files

### Resume Interrupted Experiments
- The script automatically skips completed experiments (checks for existing CSV files)
- Simply re-run the same command to continue
