import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
fontn = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
        }
data50 = np.loadtxt('cost_his50b1.txt')
data50n = np.loadtxt('cost_his50bn1.txt')
data50nn = np.loadtxt('cost_his50bnn.txt')

fig = plt.figure()
# ax = fig.add_subplot(111)
ax = brokenaxes(ylims=[(0, 6), (12, 17)], hspace= 0.05,despine=False)
ax.plot(np.arange(len(data50)), data50,c = 'red', label = 'Prioritized-DDQN+Batch', alpha=1)
ax.plot(np.arange(len(data50n)), data50n,c = 'indianred',label = 'DDQN+Batch', alpha=0.5)
ax.plot(np.arange(len(data50nn)), data50nn/32,c = 'lightcoral',label = 'DDQN', alpha=0.3)
# ax.scatter(wr1,wr2, s=30, c='g',marker='^', alpha=0.9)
# ax.scatter(dr1,dr2, s=30, c='b',marker='^', alpha=0.9)
# ax.scatter(qr1,qr2, s=30, c='r',marker='^', alpha=0.9)
# ax.scatter(cr1,cr2, s=30, c='k',marker='^', alpha=0.9)
# ax.scatter(br1,br2, s=30, c='y',marker='^', alpha=0.9)

#ax.legend(['Prioritized-DDQN+Batch','DDQN+Batch','DDQN'],prop=fontn,loc='best')
ax.legend(prop=fontn)
ax.set_ylabel('Loss of Training',fontsize=12, family='Times New Roman')
ax.set_xlabel('Training Step',fontsize=12, family='Times New Roman')

plt.savefig("jietu2.png", dpi=600, bbox_inches ='tight')
plt.show()