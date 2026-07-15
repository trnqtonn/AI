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

c1 = int(input("Cot can doi thu nhat: "))
c2 = int(input("Cot can doi thu hai: "))

A[:, [c1, c2]] = A[:, [c2, c1]]

print("\nMa tran sau khi doi: ")
print(A)
