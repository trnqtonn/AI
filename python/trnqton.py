import qrcode

image = qrcode.make("https://facebook.com/trn.qtonn")
image.save("qrcode.png",  "PNG")