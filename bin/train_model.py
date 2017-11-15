from training_gyro import Software
# from new_serial import Serial
# from client import Client
s = Software(150, True, 'log/results_new.txt')

#s.inputModule('train/1wavehands1.csv')
s.inputModule('datasets/merged/merged.csv')

# while True:
#     result = serial.run()
#     result_2_server = s.predict(result)
#     client.send(result_2_server)
