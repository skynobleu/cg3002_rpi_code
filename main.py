from Comm import comm
import serial

s = comm(serial.Serial("/dev/ttyS0",9600))

s.handshake()

while True:
	error = s.receive()
	print(error/100)
