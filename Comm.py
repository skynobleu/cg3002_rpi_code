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
						
						xa1 = result.split('|')[1]
						ya1 = result.split('|')[2]
						za1 = result.split('|')[3]
						xg1 = result.split('|')[4]
						yg1 = result.split('|')[5]
						zg1 = result.split("|")[6]
						xa2 = result.split('|')[7]
						ya2 = result.split('|')[8]
						za2 = result.split('|')[9]
						xg1 = result.split('|')[10]
						yg2 = result.split('|')[11]
						zg2 = result.split('|')[12]
						current = result.split('|')[13]
						voltage = result.split('|')[14]
						checkbit = result.split('|')[15]
						inspower = float(current)*float(voltage)
						cumpower = self.cumpower + inspower
						#print(ord(checkbit))
						if ord(checkbit) == '\n':
								checkbit == chr('#')
						if checksum == ord(checkbit):
								list1 = [int(xa1),int(ya1),int(za1),int(xg1),int(yg1),int(zg1),int(xa2),int(ya2),int(za2),int(xg2),int(yg2),int(zg2)]
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
								
								if count == 150:
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
					
