'''
Dirty Experiments

nothing really meainingful here
'''


from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
import random
from random import randint as ri
import torch
x, y, z= symbols('x y z')
sym = x, y, y #x, y, z in case of three variables... or change the parameters below

def random_function(n = 5, deg = 2, cr = 1, case_add = 6):
    f = 1
    for _ in range(n):
        if ri(0, case_add):
            f = f + ri(-cr,cr) * sym[ri(0, 2)]**ri(1, deg)
        else:
            f = (f)*(ri(1, cr)*sym[ri(0, 2)]**ri(1, deg))
    
    return expand(f)

model = torch.load("model.dat")
model.device = torch.device("cuda:0")

cnt = 0
tot = 200

for _ in range(tot):
    f1, f2 = random_function(n = 3), random_function(n = 2)
    f3 = f1**2 + f2**2
    f4 = expand(f3)
    
    
    src = str(f4).replace(' ', '').replace('**', '^')
    tgt = str(f3).replace(' ', '').replace('**', '^')
    
    print(f'Real : {tgt}')
    for i in range(1, 11):
        rets = model.toSOP(src)[0]
        res = "False"
        try:
            trs = standard_transformations + (implicit_multiplication,)
            ex = parse_expr(rets.replace('^', '**'), transformations = trs)
            
            if expand(tgt) == expand(ex):
                res = "True"
                cnt += 1
                print(f'Prediction {i} : {rets} : {res}')
                break
                
        except:
            pass
        

        print(f'Prediction {i} : {rets} : {res}')
    
print(f"{100 * cnt/tot :.2f} % are correct, with 10 possible trials.")