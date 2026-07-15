import numpy as np
a = np.arange(0, 32, 2)
b = a.reshape(4, 4)
print("Ma tran 4x4:")
print(b)
tong = np.sum(b[1])
print("Tong hang " + str(2) + "=", tong)