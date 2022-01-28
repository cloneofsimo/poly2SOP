'''
Data generation for poly2SOP.

Written by:
    Simo Ryu
'''
from sympy import *
from random import randint


def create_some_of_square(coef_range : int = 1, n_data : int = 100000):
    assert coef_range > 0

    x, y, z= symbols('x y z')
    sym = x, y, y #x, y, z in case of three variables... or change the parameters below

    pol = [x, y, 1, x*y, x*x, y*y]

    def random_polynomial(coef_range :int):
        f = 0
        for mo in pol:
            f = f + mo*randint(-coef_range, coef_range)
        return expand(f)

    FILE_x = open("train_x.txt", 'w')
    FILE_y = open("train_y.txt", 'w')

    for _ in range(n_data):
        f1, f2 = random_polynomial(coef_range), random_polynomial(coef_range)
        f3 = f1**2 + f2**2
        f4 = expand(f3)
        FILE_x.write(str(f4).replace(' ', '').replace('**', '^') + '\n')
        FILE_y.write(str(f3).replace(' ', '').replace('**', '^') + '\n')

    FILE_x.close()
    FILE_y.close()

if __name__ == "__main__":
    create_some_of_square(coef_range = 1, n_data = 100000)