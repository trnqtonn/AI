import numpy as np
m = int(input("Nhap so hang m: "))
n = int(input("Nhap so cot n: "))
A = np.zeros((m, n), dtype= int)
print("Nhap ma tran:")
for i in range(m):
    for j in range(n):
        A[i][j] = int(input(f"A[{i}][{j}] = "))
print("\nMa tran ban dau: ")
print(A)

c = int(input("Hang can tinh tong: "))
tong = np.sum(A[c])
print("Tong hang " + str(c) + "=", tong)