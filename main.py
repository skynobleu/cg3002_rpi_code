from Comm import comm
from client import client
import serial
from bin.prediction import Software
import time
classifier = Software("bin/models/KNN_new.sav")

s = comm(serial.Serial("/dev/ttyS0",115200))
c = client("192.168.1.20", 8800)
s.handshake()
previous = "started"
reset = False
prevTime = 0
cumpower = 0.0

while True:
    currTime = time.time()
    dataList = s.receive()
    preData = dataList[3]
    #predicting function here
    actnum = classifier.predictDanceMove(preData)

    print(actnum)
    print("previous: "+ str(previous) )
    if actnum != "Standing":
        current = float(dataList[0]) / 1000.0
        voltage = dataList[1]
        inspower = "{:.3f}".format(dataList[2])
        cumpower += float("{:.3f}".format(float(currTime - prevTime) * float(inspower)))
        if previous != "started":
            if previous == actnum:
                print("send: " + actnum)
                c.clientsend(actnum,voltage,current,inspower,str(cumpower))
                previous = "resetted"
                reset = True
        #else:
            #previous = actnum
    if not reset:
        previous = actnum
    else:
        reset = False

    prevTime = currTime
