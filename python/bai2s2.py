import numpy as np
a = np.arange(0, 60, 5)
print("Mang 1 chieu:")
print(a)
b = a.reshape(3, 4)
print("Mang 2 chieu:")
print(b)
print("Ma tran chuyen vi:")
print(b.T)
