import serial

ser = serial.Serial('/dev/ttyS0', 115200)

while True:
    print(ser.read())
