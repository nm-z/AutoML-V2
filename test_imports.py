import sys
print(f"Python version: {sys.version}")
try:
    import autogluon.tabular as ag
    print(f"✓ AutoGluon: {ag.__version__}")
except ImportError as e:
    print(f"✗ AutoGluon: {e}")
try:
    import tpot
    print(f"✓ TPOT: {tpot.__version__}")
except ImportError as e:
    print(f"✗ TPOT: {e}")
try:
    import autosklearn.regression
    import autosklearn
    print(f"✓ Auto-Sklearn: {autosklearn.__version__}")
except ImportError as e:
    print(f"✗ Auto-Sklearn: {e}") 