import numpy as np
a = np.arange(1, 26, 1)
b = a.reshape(5, 5)
print("Ma tran 5x5:")
print(b)
c = int(input("Cot can tinh tong: "))
tong = np.sum(b[:, c])
print("Tong hang " + str(c) + "=", tong)