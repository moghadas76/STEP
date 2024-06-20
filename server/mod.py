import importlib
import sys
print(sys.path)
try:
    importlib.import_module('STEP.basicts')
except Exception as e:
    print("1", e)

try:
    importlib.import_module('basicts')
except Exception as e:
    print("2", e)

try:
    importlib.import_module('STEP', "basicts")
except Exception as e:
    print("3", e)


try:
    from basicts.utils import load_pkl
except Exception as e:
    print("4", e)

try:
    from STEP.basicts.utils import load_pkl
except Exception as e:
    print("44", e)

try:
    import basicts
except Exception as e:
    print("5", e)