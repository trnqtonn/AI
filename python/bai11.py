import numpy as np
A = np.array([
    [int(input("A[0][0] = ")), int(input("A[0][1] = ")), int(input("A[0][2] = "))],
    [int(input("A[1][0] = ")), int(input("A[1][1] = ")), int(input("A[1][2] = "))],
    [int(input("A[2][0] = ")), int(input("A[2][1] = ")), int(input("A[2][2] = "))]
])
print("Ma tran ban dau: ")
print(A)

A[:, [0, 2]] = A[:, [2, 0]]
print("Ma tran sau doi cot: ")
print(A)