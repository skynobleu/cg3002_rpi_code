from learning import Software
# from new_serial import Serial
# from client import Client
s = Software(True, 120, 'log/results.txt')

#s.inputModule('train/1wavehands1.csv')
s.inputModule('train/merged.csv')

# while True:
#     result = serial.run()
#     result_2_server = s.predict(result)
#     client.send(result_2_server)