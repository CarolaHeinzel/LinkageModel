from sympy import symbols, simplify, Matrix
from sympy import  exp
import matplotlib.pyplot as plt
import numpy as np
from sympy import factor
from sympy import solve

# Calculates the constraints for which det(M_1) = 0
q1, p11, p21, f = symbols('q1 p11 p21 f')

a = (1 - q1*p11)*(q1 + f*(1 - q1)) + (1 - p21 + q1*p21)*(1 - q1)*(1 - f)
b = q1*p11*(q1*(1 - f) + f) + p21*(1 - q1)**2*(1 - f)
c = (1 - q1*p11)*(1 - f)*(1 - q1) + (1 - p21 + q1*p21)*(1 - q1 +q1* f)
d = q1*p11*(1 - f)*(1 - q1) + p21*(1 - q1)*(1-q1 + f*q1)

M1 = Matrix([[a, b], [c, d]])

det = simplify(M1.det())
det

det_factored = factor(det)
det_factored


factor_second = 2*f**2*q1**2 - 3*f**2*q1 + f**2 - 4*f*q1**2 + 6*f*q1 - 3*f + 2*q1**2 - 3*q1

factor_second_factored = factor(factor_second)
factor_second_factored



q1_solutions = solve(factor_second, q1)
q1_solutions



x_vals = np.linspace(40, 100, 200)
y = (np.exp(-x_vals)-3) /(2*(-1+np.exp(-x_vals)))

plt.figure(figsize=(8,6))
plt.plot(x_vals, y)
plt.xlabel('q1')
plt.ylabel('x')
plt.grid(True)
plt.show()
#%%
# Calculates the constraints for which det(M_2) = 0
from sympy import symbols, Matrix, simplify, factor

q1, p13, p23, f = symbols('q1 p13 p23 f')

a2 = (1 - q1*p13)*(f + q1*(1 - f)) + (1 - (1 - q1)*p23)*q1*(1 - f)
b2 = q1*p13*(f + q1*(1 - f)) + (1 - q1)*p23*q1*(1 - f)
c2 = (1 - q1*p13)*(1 - q1)*(1 - f) + (1 - (1 - q1)*p23)*(f + (1 - q1)*(1 - f))
d2 = q1*p13*(1 - q1)*(1 - f) + (1 - q1)*p23*(f + (1 - q1)*(1 - f))

M2 = Matrix([[a2, b2], [c2, d2]])

det_M2 = simplify(M2.det())

det_M2_factored = factor(det_M2)
det_M2_factored
