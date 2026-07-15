import numpy as np
m = int(input("Nhap so hang m: "))
n = int(input("Nhap so cot n: "))
A = np.zeros((m, n), dtype= int)
print("Nhap ma tran:")
for i in range(m):
    for j in range(n):
        A[i][j] = int(input(f"A[{i}][{j}] = "))
print("Ma tran A: ")
print(A)

B = np.zeros((m, n), dtype= int)
print("Nhap ma tran:")
for i in range(m):
    for j in range(n):
        B[i][j] = int(input(f"B[{i}][{j}] = "))
print("Ma tran B: ")
print(B)

C = np.dot(A, B)
print("\nTich hai ma tran la: ")
print(C)