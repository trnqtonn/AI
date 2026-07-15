n = int(input("Nhap so hang n = "))
so = 1
for i in range(1, n + 1):
    for j in range(i):
        print(so, end=" ")
        so += 1
    print()