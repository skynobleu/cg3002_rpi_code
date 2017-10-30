from Comm import comm
from client import client
import serial
from bin.prediction import Software
import time
classifier = Software("bin/models/KNN.sav")

s = comm(serial.Serial("/dev/ttyS0",115200))
c = client("192.168.1.20", 8888)
s.handshake()

while True:
    dataList = s.receive()
    preData = dataList[4]
    #predicting function here
    actnum = classifier.predictDanceMove(preData)
    print(actnum)
    current = dataList[0]
    voltage = dataList[1]
    inspower = "{:.3f}".format(dataList[2])
    cumpower = "{:.3f}".format(dataList[3])
    c.clientsend(actnum,voltage,current,inspower,cumpower)
