import serial
import csv

class comm:
	readyToReceive = 0
	flag = 0
	cumpower = 0
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
										#open('test.csv', 'w')
					
					

	def receive(self):
		#print("receiving")
		count = 0
		checksum = 0
		error = 0
		list2 = []
		while self.readyToReceive == 1:
				#try:
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
						inspower = float(current)*float(voltage)
						cumpower = self.cumpower + inspower
						#print(ord(checkbit))
						if ord(checkbit) == '\n':
								checkbit == chr('#')
						if checksum == ord(checkbit):
								list1 = [int(x1),int(y1),int(z1),int(x2),int(y2),int(z2)]
								#print(list1)
								#with open('test.csv', 'a') as f:
										#writer = csv.writer(f)
										#writer.writerows(list1)
										#count = count + 1
								#print(x1)
								#print(y1)
								#print(z1)
								#print(x2)
								#print(y2)
								#print(z2)
								#print(current)
								#print(voltage)
								#print(checkbit)
								list2.append(list1)
								count = count + 1
								
								if count == 200:
									list3 = []
									list3.append(current)
									list3.append(voltage)
									list3.append(inspower)
									list3.append(cumpower)
									list3.append(list2)
									return list3
				#except:
						#print("Error occurred")
						#error = error + 1
					
