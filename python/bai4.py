n = int(input("Nhap n = "))
sum = 0
for i in range(1, n):
    if n % i == 0:
        sum += i
if sum == n:
    print(str(n) + " la so hoan hao")
else:
    print(str(n) + " la so khong hoan hao")