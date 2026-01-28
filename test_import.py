#!/usr/bin/env python
"""Quick test to verify imports work."""
print("Testing imports...")

try:
    import nevo
    print("  nevo: OK")
except Exception as e:
    print(f"  nevo: FAILED - {e}")

try:
    from nevo.examples.benchmark_experiment import run_benchmark
    print("  benchmark_experiment: OK")
except Exception as e:
    print(f"  benchmark_experiment: FAILED - {e}")

try:
    import nengo
    print(f"  nengo: OK (version {nengo.__version__})")
except Exception as e:
    print(f"  nengo: FAILED - {e}")

try:
    import nengo_dl
    print(f"  nengo_dl: OK (version {nengo_dl.__version__})")
except Exception as e:
    print(f"  nengo_dl: NOT INSTALLED - {e}")

print("\nAll basic imports completed.")
