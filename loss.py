import os
import matplotlib.pyplot as plt
import numpy as np
 
MainDir = '/home/gaohan/Project/darknet'
TrainLogPath = os.path.join(MainDir, 'backup', 'TrainLog_2019-12-31-20-11.txt')
Loss, AgvLoss = [], []
with open(TrainLogPath, 'r') as FId:
	TxtLines = FId.readlines()
	for TxtLine in TxtLines:
		SplitStr = TxtLine.strip().split(',')
		Loss.append(float(SplitStr[0]))
		AgvLoss.append(float(SplitStr[1]))
 
IterNum = len(AgvLoss)
StartVal, EndVal, Stride = 1000, IterNum, 50 #视情况修改
Xs = np.arange(StartVal, EndVal, Stride)
Ys = np.array(AgvLoss[StartVal:EndVal:Stride])
plt.plot(Xs, Ys,label='avg_loss')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Loss-Iter curve")
plt.legend()
plt.show()
