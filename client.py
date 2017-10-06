# client.py  
import socket
from Crypto.Cipher import AES
import base64
import sys
import os
from Crypto import Random

# create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
actions = ['busdriver', 'frontback', 'jumping', 'jumpingjack', 'sidestep',
           'squatturnclap', 'turnclap', 'wavehands', 'windowcleaner360',
           'windowcleaning']
reading = '2'

# get local machine name
host = socket.gethostname()                           

port = 8888

# connection to hostname on the port.
s.connect((host, port))                               

reading = str(reading)
msg = '#' + 'logout  ' + '|' + reading + '|' + reading + '|' + reading + '|' + reading
length = 16 - (len(msg) % 16)
msg += length * ' '
secret_key = '1234567890abcdef'
iv = Random.new().read(AES.block_size)
cipher = AES.new(secret_key,AES.MODE_CBC,iv)
encoded = base64.b64encode(iv + cipher.encrypt(msg))
print(msg)

# Receive no more than 1024 bytes
s.send(encoded)                                    

s.close()

#print("The time got from the server is %s" % tm.decode('ascii'))
