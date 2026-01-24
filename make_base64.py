import base64

mp3_file = "sample voice 1.mp3"   # change this filename to your mp3 name

with open(mp3_file, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

print(b64)
