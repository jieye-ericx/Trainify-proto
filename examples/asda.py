import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
aa = os.path.split(rootPath)[0]
print(os.path.split(aa)[0])

print()
print(os.path.join(os.path.expandvars('$HOME'), 'Trainify_data'))

print
os.path.expanduser('~')

# print(os.environ)
