import random
import numpy as np
mydict = {}
for i in range(10):
    mydict[i] = random.randint(1, 10)
print("Tu dien: ")
print(mydict)

A = np.array(list(mydict.items()))
print("Mang 2 chieu: ")
print(A)