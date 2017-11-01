from Comm import comm
from client import client
import serial
from bin.prediction import Software
import time
classifier = Software("bin/models/KNN.sav")

s = comm(serial.Serial("/dev/ttyS0",115200))
c = client("192.168.1.233", 8888)
s.handshake()
previous = "started"
reset = False
while True:
    dataList = s.receive()
    preData = dataList[4]
    #predicting function here
    actnum = classifier.predictDanceMove(preData)

    print(actnum)
    print("previous: "+ str(previous) )
    if actnum != "Standing":
        current = dataList[0]
        voltage = dataList[1]
        inspower = "{:.3f}".format(dataList[2])
        cumpower = "{:.3f}".format(dataList[3])
        if previous != "started":
            if previous == actnum:
                print("send: " + actnum)
                c.clientsend(actnum,voltage,current,inspower,cumpower)
                previous = "resetted"
                reset = True
        #else:
            #previous = actnum
    if not reset:
        previous = actnum
    else:
        reset = False
