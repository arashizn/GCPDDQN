import numpy as np
import matplotlib.pyplot as plt
fontn = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
        }
data25 = np.loadtxt('his_prio25.txt')
data50 = np.loadtxt('his_prio505.txt')
data100 = np.loadtxt('his_prio100.txt')
step25 = data25[1,:]
step50 = data50[1,:]
step100 = data100[1,:]
fig = plt.figure()
ax = fig.add_subplot(111)
#f,ax = plt.subplots(1,1,figsize=(10,8))
# ax=axs[0]
# ax2=axs[1]

ax.plot(np.arange(500), np.ones(500)*28,c = 'b',alpha=0.6,linestyle='dashed')
ax.plot(np.arange(500), np.ones(500)*20,c = 'g',alpha=0.6,linestyle='dashed')
ax.plot(np.arange(500), np.ones(500)*19,c = 'k',alpha=0.6,linestyle='dashed')
ax.plot(np.arange(500), np.ones(500)*21,c = 'y',alpha=0.6,linestyle='dashed')
ax.plot(np.arange(500), np.ones(500)*min(step50),c = 'r',alpha=0.6,linestyle='dashed')
ax.plot(np.arange(500), step50,c = 'r',alpha=0.6)
# ax.set_title('UAV Num = 50',fontn)
tick = np.linspace(15,40,6)
ax.set_yticks(tick)
ax.legend(['Weight','Degree','Closeness','Betweenness','GCPDDQN'],prop=fontn,loc='best')

ax.set_ylabel('Totle Step',fontn)
ax.set_xlabel('Episode',fontn)
# ax2 = plt.subplot(122)

# ax2.plot(np.arange(500), np.ones(500)*61,c = 'b',alpha=0.6,linestyle='dashed')
# ax2.plot(np.arange(500), np.ones(500)*46,c = 'g',alpha=0.6,linestyle='dashed')
# ax2.plot(np.arange(500), np.ones(500)*min(step100),c = 'r',alpha=0.6,linestyle='dashed')
# ax2.plot(np.arange(500), np.ones(500)*35,c = 'k',alpha=0.6,linestyle='dashed')
# ax2.plot(np.arange(500), np.ones(500)*33,c = 'y',alpha=0.6,linestyle='dashed')
# ax2.plot(np.arange(500), step100,c = 'r',alpha=0.6)
# ax2.set_title('UAV Num = 100',fontn)
# tick2 = np.linspace(30,80,6)
# ax2.set_yticks(tick2)
# ax2.legend(['Weight','Degree','DQN','Closeness','Betweenness'],prop=fontn,loc='best')
# ax2.set_ylabel('Totle Step',fontn)
# ax2.set_xlabel('Episode',fontn)
plt.savefig("jietu1.png", dpi=600, bbox_inches ='tight')
plt.show()