import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(__file__)
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))

print()

print(os.path.split(os.path.realpath(__file__)))
sys.path.append(BASE_DIR)

print('-----')
PROJECT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(DATA_DIR_PATH)
