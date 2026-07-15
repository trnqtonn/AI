ds = []
tong = 0
while True:
    x = input("Nhap gia tri : ")
    if x == "$":
        break
    ds.append(float(x))
for i in ds:
    tong += i
tb = tong / len(ds)
print("Trung binh =", tb)