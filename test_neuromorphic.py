#!/usr/bin/env python
"""
Simple test script to verify neuromorphic operators work.
Run this to check if spike decoding is working.
"""

import numpy as np
from nevo import NEVOptimiser

def sphere(x):
    return float(np.sum(x ** 2))

print("Testing neuromorphic dual mode...")
print("=" * 60)

try:
    opt = NEVOptimiser(
        objective_function=sphere,
        bounds=(-5, 5),
        dimension=5,
        population_size=20,
        memory_size=10,
        operator_mode="neuromorphic_dual",
        seed=42
    )

    print(f"Operators: {[op.name for op in opt.operators]}")
    print(f"Operators have build_network: {[hasattr(op, 'build_network') for op in opt.operators]}")
    print(f"Operators use_numpy_fallback: {[op.use_numpy_fallback for op in opt.operators]}")
    print()

    print("Running optimization for 0.5 seconds...")
    opt.run(time=0.5, verbose=False)

    print()
    print(f"✅ SUCCESS!")
    print(f"Best fitness: {opt.state['best_f']:.6e}")
    print(f"Total evaluations: {opt.state['total_evals']}")

except Exception as e:
    print()
    print(f"❌ ERROR: {type(e).__name__}")
    print(f"Message: {e}")
    import traceback
    traceback.print_exc()

