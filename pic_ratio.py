import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from uavs_env import UAVSEnv
fontn = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
        }

env = UAVSEnv()
# g_d = nx.read_gpickle('test100ba.gpickle')
# g_w = nx.read_gpickle('test100ba.gpickle')
# g_q = nx.read_gpickle('test100ba.gpickle')
# g_c = nx.read_gpickle('test100ba.gpickle')
# g_b = nx.read_gpickle('test100ba.gpickle')
g_d = nx.read_gpickle('test100.gpickle')
g_w = nx.read_gpickle('test100.gpickle')
g_q = nx.read_gpickle('test100.gpickle')
g_c = nx.read_gpickle('test100.gpickle')
g_b = nx.read_gpickle('test100.gpickle')
si = env.COOPScore(g_d)
degree = [93, 15, 38, 82, 0, 5, 11, 27, 46, 53, 58, 66, 83, 1, 9, 32, 48, 54, 70, 76, 88, 92, 12, 16, 20, 23, 34, 42, 61, 71, 77, 98, 26, 30, 43, 63, 72, 78, 86, 94, 99, 3, 7, 17, 21, 37, 47]
weight = [45, 75, 64, 6, 78, 69, 0, 19, 28, 15, 85, 20, 90, 13, 39, 97, 96, 40, 82, 36, 32, 60, 22, 37, 23, 2, 63, 34, 99, 52, 41, 94, 56, 43, 73, 67, 51, 54, 65, 18, 88, 55, 25, 9, 27, 50, 24, 95, 3, 81, 87, 68, 77, 72, 21, 17, 46, 31, 84, 76, 74, 11]
dqn = [15, 82, 90, 64, 99, 47, 58, 6, 23, 20, 75, 19, 27, 77, 43, 69, 48, 96, 46, 76, 54, 25, 68, 34, 93, 11, 0, 79, 32, 38, 39, 5, 53, 36, 98, 12, 94, 80]
closeness = [93, 38, 50, 80, 98, 12, 35, 47, 9, 66, 25, 77, 0, 27, 83, 99, 39, 68, 23, 75, 58, 32, 54, 57, 44, 11, 76, 19, 88, 89, 40, 13, 90, 5, 64, 53]
betweenness = [93, 90, 98, 80, 82, 35, 47, 15, 66, 77, 9, 20, 68, 64, 94, 95, 12, 19, 7, 5, 3, 4, 27, 58, 99, 33, 26, 46, 54, 34, 57, 81, 48, 42]
# degree = [3, 0, 2, 5, 6, 10, 8, 17, 14, 20, 4, 11, 19, 27, 70, 15, 22, 49, 23, 29, 33, 37, 66, 9, 31, 36]
# weight = [68, 49, 71, 7, 97, 85, 79, 41, 39, 42, 64, 15, 98, 69, 35, 60, 54, 95, 73, 63, 83, 96, 92, 1, 75, 55, 65, 31, 37, 44, 57, 84, 30, 59, 58, 25, 90, 86, 34, 74, 81, 27, 4, 53, 36, 18, 78, 62, 93, 13, 2, 6, 88, 99, 20, 77, 87, 12, 67, 72, 82, 80, 38, 23, 10, 16, 43, 0]
# dqn = [2, 3, 0, 6, 5, 10, 49, 4, 66, 15, 11, 31, 8, 29, 37, 14, 17, 51, 33, 19, 23, 27, 20, 70]
# closeness = [3, 2, 0, 5, 10, 14, 6, 25, 37, 19, 20, 11, 27, 22, 70, 4, 66, 15, 49, 8, 31, 39, 36, 33, 69, 17, 62, 48]
# betweenness = [3, 2, 0, 5, 10, 6, 8, 20, 17, 49, 19, 70, 14, 4, 15, 22, 11, 66, 29, 23, 27, 36, 52, 33, 37]

dr=[]
dr1= []
dr2=[]
wr=[]
wr1=[]
wr2= []
qr=[]
qr1= []
qr2=[]

cr=[]
cr1= []
cr2=[]
br=[]
br1= []
br2=[]
dr.append(1)
wr.append(1)
qr.append(1)
cr.append(1)
br.append(1)
for i in range(len(degree)):
    g_d.remove_node(degree[i])
    sub = env.subgraph(g_d)
    sub_coop = [env.COOPScore(g) for g in sub]
    ratio = max(sub_coop)/si

    dr.append(ratio)
for i in range(len(weight)):
    g_w.remove_node(weight[i])
    sub = env.subgraph(g_w)
    sub_coop = [env.COOPScore(g) for g in sub]
    ratio = max(sub_coop)/si
    wr.append(ratio)
for i in range(len(dqn)):
    g_q.remove_node(dqn[i])
    sub = env.subgraph(g_q)
    sub_coop = [env.COOPScore(g) for g in sub]
    ratio = max(sub_coop)/si
    qr.append(ratio)
for i in range(len(closeness)):
    g_c.remove_node(closeness[i])
    sub = env.subgraph(g_c)
    sub_coop = [env.COOPScore(g) for g in sub]
    ratio = max(sub_coop)/si
    cr.append(ratio)

for i in range(len(betweenness)):
    g_b.remove_node(betweenness[i])
    sub = env.subgraph(g_b)
    sub_coop = [env.COOPScore(g) for g in sub]
    ratio = max(sub_coop)/si
    br.append(ratio)

exist = ( np.array(wr) < 0.7) * 1
index = np.argmax(exist)
wr1.append(index)
wr2.append(wr[index])
exist = ( np.array(wr) < 0.5) * 1
index = np.argmax(exist)
wr1.append(index)
wr2.append(wr[index])
exist = ( np.array(wr) < 0.3) * 1
index = np.argmax(exist)
wr1.append(index)
wr2.append(wr[index])

exist = ( np.array(dr) < 0.7) * 1
index = np.argmax(exist)
dr1.append(index)
dr2.append(dr[index])
exist = ( np.array(dr) < 0.5) * 1
index = np.argmax(exist)
dr1.append(index)
dr2.append(dr[index])
exist = ( np.array(dr) < 0.3) * 1
index = np.argmax(exist)
dr1.append(index)
dr2.append(dr[index])


exist = ( np.array(qr) < 0.7) * 1
index = np.argmax(exist)
qr1.append(index)
qr2.append(qr[index])
exist = ( np.array(qr) < 0.5) * 1
index = np.argmax(exist)
qr1.append(index)
qr2.append(qr[index])
exist = ( np.array(qr) < 0.3) * 1
index = np.argmax(exist)
qr1.append(index)
qr2.append(qr[index])


exist = ( np.array(cr) < 0.7) * 1
index = np.argmax(exist)
cr1.append(index)
cr2.append(cr[index])
exist = ( np.array(cr) < 0.5) * 1
index = np.argmax(exist)
cr1.append(index)
cr2.append(cr[index])
exist = ( np.array(cr) < 0.3) * 1
index = np.argmax(exist)
cr1.append(index)
cr2.append(cr[index])

exist = ( np.array(br) < 0.7) * 1
index = np.argmax(exist)
br1.append(index)
br2.append(br[index])
exist = ( np.array(br) < 0.5) * 1
index = np.argmax(exist)
br1.append(index)
br2.append(br[index])
exist = ( np.array(br) < 0.3) * 1
index = np.argmax(exist)
br1.append(index)
br2.append(br[index])
print(br1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(len(weight)+1), wr,c = 'g',alpha=0.6)
ax.plot(np.arange(len(degree)+1), dr,c = 'b',alpha=0.6)
ax.plot(np.arange(len(closeness)+1), cr,c = 'k',alpha=0.6)
ax.plot(np.arange(len(betweenness)+1), br,c = 'y',alpha=0.6)
ax.plot(np.arange(len(dqn)+1), qr,c = 'r',alpha=0.6)
ax.scatter(wr1,wr2, s=30, c='g',marker='^', alpha=0.9)
ax.scatter(dr1,dr2, s=30, c='b',marker='^', alpha=0.9)
ax.scatter(cr1,cr2, s=30, c='k',marker='^', alpha=0.9)
ax.scatter(br1,br2, s=30, c='y',marker='^', alpha=0.9)
ax.scatter(qr1,qr2, s=30, c='r',marker='^', alpha=0.9)

ax.legend(['Weight','Degree','Closeness','Betweenness','GCPDDQN'],prop=fontn,loc='best')
ax.set_ylabel('COOP Ratio of the maximum subgraph ',fontn)
ax.set_xlabel('Step of removal node',fontn)

tick = np.linspace(0,1,6)
ax.set_yticks(tick)
for i in range(3):
    plt.vlines(wr1[i], 0, wr2[i], colors='g', linestyle="dotted",alpha =0.6)
    # plt.text(wr1[i], 0, '%d' % wr1[i],verticalalignment='bottom', fontsize =8)
    plt.vlines(dr1[i], 0, dr2[i], colors='b', linestyle="dotted",alpha =0.6)
    # plt.text(dr1[i], 0, '%d' %dr1[i],verticalalignment='bottom',fontsize =8)
    plt.vlines(qr1[i], 0, qr2[i], colors='r', linestyle="dotted",alpha =0.6)
    # plt.text(qr1[i], 0, '%d' %qr1[i],verticalalignment='bottom',fontsize =8)
    plt.vlines(cr1[i], 0, cr2[i], colors='k', linestyle="dotted",alpha =0.6)
    # plt.text(cr1[i], 0, '%d' %cr1[i],verticalalignment='bottom',fontsize =8)
    plt.vlines(br1[i], 0, br2[i], colors='y', linestyle="dotted",alpha =0.6)
    # plt.text(br1[i], 0, '%d' %br1[i],verticalalignment='bottom',fontsize =8)

plt.ylim(0,None)
plt.savefig("ratio.png", dpi=600, bbox_inches ='tight')
plt.show()