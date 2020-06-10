import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


#sampFreq, snd = wavfile.read('aa.wav')
sampFreq, snd = wavfile.read('440_sine.wav')

print(snd.dtype, "\n")
snd = snd / (2.**15)

print(5060 / sampFreq, "\n")

s1 = snd[:, 0]

#timeArray = np.arange(0, 5060.0, 1)
timeArray = np.arange(0, len(snd), 1)
timeArray = timeArray / sampFreq
print(timeArray)

#plt.plot(timeArray, s1, color='black')
plt.plot(timeArray, s1, color='k')
plt.ylabel('Amplitude')
plt.xlabel('Time (ms)')

plt.show()

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

#绘制的频谱图如下所示。注意图中y轴是能量的对数10*log10(p)，单位分贝；x轴是频率/1000，单位kHz。
freqArray = np.arange(0, nUniquePts, 1.0) * (sampFreq / n)
plt.plot(freqArray / 1000, 10 * np.log10(p), color='k')
plt.xlabel('Freqency (kHz)')
plt.ylabel('Power (dB)')
plt.show()

#为了检验计算结果是否等于信号的能量，我们计算出信号的均方根rms。广义来说，可以用rms衡量波形的幅度。如果直接对偏移量为零的正弦波求幅度的均值，它的正负部分相互抵消，结果为零。那我们先对幅度求平方，再开方（注意：开方加大了幅度极值的权重？）
rms_val = np.sqrt(np.mean(s1**2))
print(rms_val)

#信号的rms等于总能量的平方根，那么把fft在所有频率上的能量值相加然后求平方根，应该等于rms
print(np.sqrt(sum(p)))







