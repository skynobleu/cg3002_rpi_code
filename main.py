from Comm import comm
from client import client
import serial
from bin.prediction import Software
import time
classifier = Software(100)
s = comm(serial.Serial("/dev/ttyS0",115200))
c = client("192.168.43.146", 8888)
s.handshake()

while True:
    dataList = s.receive()
    preData = dataList[4]
    #predicting function here
    actnum = classifier.predictDanceMove(preData)
    current = dataList[0]
    voltage = dataList[1]
    inspower = dataList[2]
    cumpower = dataList[3]
    c.clientsend(actnum,voltage,current,inspower,cumpower)
