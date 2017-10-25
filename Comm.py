import serial
import csv

class comm:
	readyToReceive = 0
	flag = 0
	def __init__(self,ser):
		self.ser = ser
		
	def handshake(self):
		#print("handshaking")
		while self.readyToReceive == 0:
				hello = int.from_bytes(self.ser.read(),byteorder = 'little')
				print(hello)
				if hello == 1:
						while(self.flag != 1):
								self.ser.write((2).to_bytes(1,byteorder = 'little'))
								rec = self.ser.read()
								rec = int.from_bytes(rec,byteorder ='little') 
								print(rec)
								if rec == 3:
										self.readyToReceive = 1
										self.flag = 1
										open('test.csv', 'w')
					
					

	def receive(self):
		#print("receiving")
		count = 0
		checksum = 0
		error = 0
		while self.readyToReceive == 1:
				try:
						read_serial = self.ser.readline()
						result = '|'
						result = result + str(read_serial,"utf-8")
						checksum = 0

						
						for i in result[1:-3]:
								checksum ^= ord(i)
						#print(checksum)
						
						x1 = result.split('|')[1]
						y1 = result.split('|')[2]
						z1 = result.split('|')[3]
						x2 = result.split('|')[4]
						y2 = result.split('|')[5]
						z2 = result.split('|')[6]
						current = result.split('|')[7]
						voltage = result.split('|')[8]
						checkbit = result.split('|')[9]
						#print(ord(checkbit))
						if checksum == ord(checkbit):
								list1 = [[x1,y1,z1,x2,y2,z2]]
								#print(list1)
								with open('test.csv', 'a') as f:
										writer = csv.writer(f)
										writer.writerows(list1)
										count = count + 1
								#print(x1)
								#print(y1)
								#print(z1)
								#print(x2)
								#print(y2)
								#print(z2)
								#print(current)
								#print(voltage)
								#print(checkbit)
								
								if count == 100:
									return error
				except:
						print("Error occurred")
						error = error + 1
					
