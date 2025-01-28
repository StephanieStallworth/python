import qrcode

# Default URL when app is run locally htpps://127.0.0.1:8000
# Need to update to public URL once app is deployed to create new QR code to be shared out
image = qrcode.make("htpps://127.0.0.1:8000")
image.save("qr.png")