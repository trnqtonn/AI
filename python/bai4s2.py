import numpy as np
A = np.array([
    [int(input("A[0][0] = ")), int(input("A[0][1] = ")), int(input("A[0][2] = "))],
    [int(input("A[1][0] = ")), int(input("A[1][1] = ")), int(input("A[1][2] = "))],
    [int(input("A[2][0] = ")), int(input("A[2][1] = ")), int(input("A[2][2] = "))]
])
v = np.array([
    int(input("v[0] = ")),
    int(input("v[1] = ")),
    int(input("v[2] = "))
])
kq = np.dot(A, v)

print("Ma tran 3x3: ")
print(A)

print("Vecto: ")
print(v)

print("Ket qua: ")
print(kq)