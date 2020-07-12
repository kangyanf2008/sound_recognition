import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


sampFreq, snd = wavfile.read('resource/car_1.wav')
#sampFreq, snd = wavfile.read('440_sine.wav')
#sampFreq, snd = wavfile.read('20200626_095844.wav')
print("采样率=",sampFreq)
print("编码字节数=",snd.dtype, "\n")
snd = snd / (2.**15)
print("采样点数=",snd.shape[0],",通道数=",snd.shape[1], "\n")
#表示文件包含2个通道，5060个采样点。结合采样率（sampFreq = 44110），可得信号持续时长为114ms：
print("采样点数/采样率=时间（",snd.shape[0] / sampFreq, "）秒\n")
print("采样点数/采样率=时间（",len(snd)/ sampFreq, "）秒\n")
#声道1
s1 = snd[:, 0]
#声道2
s2 = snd[:, 1]
#timeArray = np.arange(0, 5060.0, 1)
timeArray = np.arange(0, len(snd), 1)
timeArray = timeArray / sampFreq
print("时间",timeArray)
#或者 np.arange(0,采样点)*（1.0/采样率）
time = np.arange(0,len(snd))*(1.0/sampFreq)
print("时间",time)

#plt.plot(timeArray, s1, color='black')
plt.plot(timeArray, s1, color='k')
plt.plot(timeArray, s2, color='c')
plt.ylabel('Amplitude')
plt.xlabel('Time (ms)')

plt.show()

#声道1
n = len(s1)
p = np.fft.fft(s1)

nUniquePts=n
#nUniquePts = np.ceil((n+1)/2.0)
#print(abs(nUniquePts))
#p = p[0:int(nUniquePts)]
p = p[0:]
print(p)

p = np.abs(p)
p = p / float(n) #除以采样点数，去除幅度对信号长度或采样频率的依赖
p = p**2 #求平方得到能量

# 乘2（详见技术手册）
# 奇nfft排除奈奎斯特点
if n % 2 > 0:  # fft点数为奇
    p[1:len(p)] = p[1:len(p)] * 2
else:  # fft点数为偶
    p[1:len(p) - 1] = p[1:len(p) - 1] * 2

#声道2
n2 = len(s1)
p2 = np.fft.fft(s1)

#nUniquePts = np.ceil((n+1)/2.0)
#print(abs(nUniquePts))
#p = p[0:int(nUniquePts)]
p2 = p2[0:]


p2 = np.abs(p2)
p2 = p2 / float(n2) #除以采样点数，去除幅度对信号长度或采样频率的依赖
p2 = p2**2 #求平方得到能量

# 乘2（详见技术手册）
# 奇nfft排除奈奎斯特点
if n2 % 2 > 0:  # fft点数为奇
    p[1:len(p2)] = p2[1:len(p2)] * 2
else:  # fft点数为偶
    p2[1:len(p2) - 1] = p2[1:len(p2) - 1] * 2

'''
print("#####################################################\n")
for index, item in enumerate(p):
    print(item)
'''

print("傅立叶后数据长度",len(p))
print("时间",len(snd)/sampFreq)
print("采样点数/采样率=时间，取整后为生样本数量",int(len(p)/sampFreq))
#样本数量
sampNum = int(len(p)/sampFreq)
#傅立叶fft变化后，每秒钟一个样本
sampData = np.ones((sampNum,2,sampFreq), dtype=np.float64)
i = 0
while i < sampNum:
    #声道1
    sampData[i][0] = p[i * sampFreq:(i + 1) * sampFreq]
    #声道2
    sampData[i][1] = p2[i * sampFreq:(i + 1) * sampFreq]
    i = i + 1

print("#####################################################\n")
print(sampData)

#绘制的频谱图如下所示。注意图中y轴是能量的对数10*log10(p)，单位分贝；x轴是频率/1000，单位kHz。
freqArray = np.arange(0, nUniquePts, 1.0) * (sampFreq / n)
plt.plot(freqArray / 1000, 10 * np.log10(p), color='k')
plt.xlabel('Freqency (kHz)')
plt.ylabel('Power (dB)')
plt.show()

#为了检验计算结果是否等于信号的能量，我们计算出信号的均方根rms。广义来说，可以用rms衡量波形的幅度。如果直接对偏移量为零的正弦波求幅度的均值，它的正负部分相互抵消，结果为零。那我们先对幅度求平方，再开方（注意：开方加大了幅度极值的权重？）
rms_val = np.sqrt(np.mean(s1**2))
print("#####################################################\n")
print(rms_val)
print("#####################################################\n")
#信号的rms等于总能量的平方根，那么把fft在所有频率上的能量值相加然后求平方根，应该等于rms
print(np.sqrt(sum(p)))







