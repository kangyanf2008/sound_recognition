import numpy as np

p = (1,2,3,4,5,6,7,8,9,10)
p2 = (11,12,13,14,15,16,17,18,19,20)
print(p)
print(p2)
sampNum = 3
#sampData = [[sampNum]*2]*sampNum
sampData = np.ones((sampNum,2,sampNum), dtype=np.int64)
i = 0
while i < 3:
    sampData[i][0] = p[i * sampNum:(i + 1) * sampNum]
    sampData[i][1] = p2[i * sampNum:(i + 1) * sampNum]
    i = i + 1

print(sampData)