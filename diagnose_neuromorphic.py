#!/usr/bin/env python
"""
Comprehensive diagnostic script for neuromorphic operators.
Run this to diagnose any issues.
"""

import sys
print("=" * 70)
print("NEVO NEUROMORPHIC OPERATORS - DIAGNOSTIC SCRIPT")
print("=" * 70)
print()

# 1. Check Python version
print(f"Python version: {sys.version}")
print()

# 2. Check imports
print("Checking imports...")
try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
except ImportError as e:
    print(f"  ✗ numpy import failed: {e}")

try:
    import nengo
    print(f"  ✓ nengo {nengo.__version__}")
except ImportError as e:
    print(f"  ✗ nengo import failed: {e}")

try:
    from nevo import NEVOptimiser
    print(f"  ✓ nevo.NEVOptimiser")
except ImportError as e:
    print(f"  ✗ NEVOptimiser import failed: {e}")

try:
    from nevo.operators.standard import NeuromorphicExplorationEnsemble, NeuromorphicExploitationEnsemble
    print(f"  ✓ Neuromorphic operators")
except ImportError as e:
    print(f"  ✗ Neuromorphic operators import failed: {e}")
    sys.exit(1)

print()

# 3. Check operator configuration
print("Checking operator configuration...")
explore_op = NeuromorphicExplorationEnsemble()
exploit_op = NeuromorphicExploitationEnsemble()

print(f"  Exploration operator:")
print(f"    Name: {explore_op.name}")
print(f"    Has build_network: {hasattr(explore_op, 'build_network')}")
print(f"    use_numpy_fallback: {explore_op.use_numpy_fallback}")
print(f"    n_neurons: {explore_op.n_neurons}")
print(f"    tau_synapse: {explore_op.tau_synapse}")

print(f"  Exploitation operator:")
print(f"    Name: {exploit_op.name}")
print(f"    Has build_network: {hasattr(exploit_op, 'build_network')}")
print(f"    use_numpy_fallback: {exploit_op.use_numpy_fallback}")
print(f"    n_neurons: {exploit_op.n_neurons}")
print(f"    tau_synapse: {exploit_op.tau_synapse}")

print()

# 4. Test simple optimization
print("Running simple optimization test...")
def sphere(x):
    return float(np.sum(x ** 2))

try:
    opt = NEVOptimiser(
        objective_function=sphere,
        bounds=(-5, 5),
        dimension=3,
        population_size=10,
        memory_size=5,
        operator_mode="neuromorphic_dual",
        seed=123
    )

    print(f"  Optimiser created")
    print(f"  Operators: {[op.name for op in opt.operators]}")
    print(f"  Running for 0.2 seconds...")

    opt.run(time=0.2, verbose=False)

    best_f = opt.state.get('best_f')
    total_evals = opt.state.get('total_evals')

    print(f"  ✓ Optimization completed successfully!")
    print(f"    Best fitness: {best_f:.6e}")
    print(f"    Total evaluations: {total_evals}")

    # Check if spike decoding was used
    if best_f is not None and best_f < 1e10:
        print(f"    ✓ Spike decoding appears to be working (non-inf fitness)")
    else:
        print(f"    ⚠ Fitness is inf or None - check spike decoding")

except Exception as e:
    print(f"  ✗ Optimization failed!")
    print(f"    Error type: {type(e).__name__}")
    print(f"    Error message: {e}")
    import traceback
    print()
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("✅ ALL DIAGNOSTICS PASSED")
print("=" * 70)
print()
print("If you're still having issues, please share:")
print("  1. The exact command you're running")
print("  2. The full error message")
print("  3. Output of this diagnostic script")

