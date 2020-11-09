from sympy import *
import random
from random import randint as ri
x, y, z= symbols('x y z')
sym = x, y, y #x, y, z in case of three variables... or change the parameters below

pol = [x, y, 1, x*y, x*x, y*y]

def random_function(cr = 1):
    f = 0
    for mo in pol:
        f = f + mo*ri(-cr, cr)
    return expand(f)
data_cases = 100000
FILE_x = open("train_x.txt", 'w')
FILE_y = open("train_y.txt", 'w')
for _ in range(data_cases):
    f1, f2 = random_function(), random_function()
    f3 = f1**2 + f2**2
    f4 = expand(f3)
    FILE_x.write(str(f4).replace(' ', '').replace('**', '^') + '\n')
    FILE_y.write(str(f3).replace(' ', '').replace('**', '^') + '\n')

FILE_x.close()
FILE_y.close()    