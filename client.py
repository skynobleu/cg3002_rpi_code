# client.py  
import socket
from Crypto.Cipher import AES
import base64
import sys
import os
from Crypto import Random

class client:

	def __init__(self, host, port):

		# get local machine name
		self.host = host
		self.port = port
		
	def clientsend(self,actnum,voltage,current,power,cumpower):
		
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
		
		actions = ['busdriver', 'frontback', 'jumping', 'jumpingjack', 'sidestep',
				   'squatturnclap', 'turnclap', 'wavehands', 'windowcleaner360',
				   'windowcleaning']



		# connection to hostname on the port.
		s.connect((self.host, self.port))

		msg = '#' + actnum + '|' + voltage + '|' + current + '|' + power + '|' + cumpower
		length = 16 - (len(msg) % 16)
		msg += length * ' '
		secret_key = '1234567890abcdef'
		iv = Random.new().read(AES.block_size)
		cipher = AES.new(secret_key,AES.MODE_CBC,iv)
		encoded = base64.b64encode(iv + cipher.encrypt(msg))
		#print(msg)

		# Receive no more than 1024 bytes
		s.send(encoded)                                    

		s.close()

		#print("The time got from the server is %s" % tm.decode('ascii'))
