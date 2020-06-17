datafile = open('op_group_testnet.txt', 'w')

start1 = 415
start2 = 417
for i in range(100):
    i += 200
    datafile.write('Conv2D_' + str(i) + ' ' + str(start1) + ' ' + str(start2) + ' 32,1,1,3 Conv2D' + '\n')
    start1 += 1
    start2 += 1
    datafile.write('dense_' + str(i+2) + ' ' + str(start1) + ' ' + str(start2) + ' 32,1,1,3 dense' + '\n')
    start1 += 1
    start2 += 1