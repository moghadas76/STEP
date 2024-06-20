import sys
print(sys.path)
try:
    from STEP.server.mod import *
except:
    print('import failed')

try:
    from mod import *
    print('imported mod')
except:
    print('import failed!!!!!!!!')


print('bod.py:', __name__, __file__)