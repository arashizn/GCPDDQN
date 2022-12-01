import networkx as nx
import numpy as np
from batch import Batchgraph
import matplotlib.pyplot as plt
from Comparemethod import Choosemethod
fontn = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
        }
g = nx.read_gpickle('test100ba.gpickle')
A = [[0.5**0.5,-0.5**0.5,0],[0.5**0.5,0.5**0.5,0],[0,0,1]]

# 3d spring layout
pos = nx.spring_layout(g, dim=3)
for v in g:
    pos[v] = np.array(pos[v])*[2000,2000,250]+[0,0,250]
# pos = pos * A
# pos = nx.kamada_kawai_layout(g, dim=3)
# Extract node and edge positions from the layout
node_xyz = np.array([pos[v] for v in g])

node_weight = np.array([g.nodes[v]['weight'] for v in g])

#c= list(nx.get_node_attributes(g,'weight').values())
edge_xyz = np.array([(pos[u], pos[v]) for u, v in g.edges()])

# Create the 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# Plot the nodes - alpha is scaled by "depth" automatically

sc=ax.scatter(*node_xyz.T, s=30, c=node_weight, linewidth =0.4,ec='w',marker='o',cmap='RdYlGn_r', alpha=0.9)
cf = plt.colorbar(sc, shrink=0.5,pad=0.0,location = 'left')
cf.ax.set_title('Lethality',fontn)

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray", linewidth =0.5, alpha = 0.8)

def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    # ax.grid(True)
    # Suppress tick labels
    tick = np.linspace(-2000,2000,6)
    tick2 = np.linspace(0,500,6)
    for dim in (ax.xaxis, ax.yaxis):
        dim.set_ticks(tick)
    ax.zaxis.set_ticks(tick2)
    # Set axes labels
    ax.set_xlabel("GroundX(m)",fontn,labelpad=0)
    ax.set_ylabel("GroundY(m)",fontn,labelpad=0)
    ax.set_zlabel("HeightZ(m)",fontn,labelpad=0)


    #刻度值字体大小设置（x轴和y轴同时设置）
    ax.tick_params(labelsize=8)
    ax.tick_params(pad = 0.03)  #通过pad参数调整距离
    ax.tick_params(grid_alpha = 0.1)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

_format_axes(ax)
fig.tight_layout()
plt.savefig("binglie5.png", dpi=600, bbox_inches ='tight')
plt.show()

g_t = []
g1 = nx.read_gpickle('test100ba.gpickle')
g2 = nx.read_gpickle('test100ba.gpickle')
g3 = nx.read_gpickle('test100ba.gpickle')
# g1 = nx.read_gpickle('test100.gpickle')
# g2 = nx.read_gpickle('test100.gpickle')
# g3 = nx.read_gpickle('test100.gpickle')
# g1.remove_nodes_from([15, 82, 90, 64, 99, 47, 58, 6, 23, 20])
# g2.remove_nodes_from([15, 82, 90, 64, 99, 47, 58, 6, 23, 20, 75, 19, 27, 77, 43, 69, 48, 96, 46, 76, 54, 25, 68, 34, 93])
# g3.remove_nodes_from([15, 82, 90, 64, 99, 47, 58, 6, 23, 20, 75, 19, 27, 77, 43, 69, 48, 96, 46, 76, 54, 25, 68, 34, 93, 11, 0, 79, 32, 38, 39, 5, 53, 36, 98, 12, 94, 80])
g1.remove_nodes_from([2, 3, 0, 6, 5, 10, 49, 4, 66, 15])
g2.remove_nodes_from([2, 3, 0, 6, 5, 10, 49, 4, 66, 15, 11, 31, 8, 29, 37,14,17,51])
g3.remove_nodes_from([2, 3, 0, 6, 5, 10, 49, 4, 66, 15, 11, 31, 8, 29, 37, 14, 17, 51, 33, 19, 23, 27, 20, 70])
g_t.append(g1)
g_t.append(g2)
g_t.append(g3)
fig = plt.figure()
for i in range(len(g_t)):
    g = g_t[i]
    sub = []
    for a in nx.connected_components(g):
        subgraph = g.subgraph(a)
        sub.append(subgraph)
    # 3d spring layout
    # pos = nx.spring_layout(g, dim=3, seed=779)
    # for v in sorted(g):
    #     pos[v] = np.array(pos[v])*[2000,2000,250]+[0,0,250]
    #pos = nx.kamada_kawai_layout(g, dim=3)
    
    node_xyz = np.array([pos[v] for v in g])
    node_weight = np.array([g.nodes[v]['weight'] for v in g])

    ax = fig.add_subplot(1,1,1, projection="3d")
    # Plot the nodes - alpha is scaled by "depth" automatically

    sc=ax.scatter(*node_xyz.T, s=30, c=node_weight, linewidth =0.4,ec='w',marker='o',cmap='RdYlGn_r', alpha=1)
    # plt.colorbar(sc, shrink=0.5, label='Weight')
    color = ['tab:gray','r','g','b','k','salmon','c','m','pink','y','cyan','lightcoral','coral','purple','greenyellow','gold']
    for j in range(len(sub)):
        print(len(sub))
        #c= list(nx.get_node_attributes(g,'weight').values())
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in sub[j].edges()])
        # sub_node_xyz = np.array([pos[v] for v in sorted(sub[j])])
        # sc=ax.scatter(*sub_node_xyz.T, s=1500, marker='o',c= color[j], alpha=0.3)

        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color=color[j], linewidth =0.5,alpha = 0.8)
    _format_axes(ax)
    plt.savefig("binglie%d.png"%(i+6), dpi=600, bbox_inches ='tight')
    # fig.tight_layout()
plt.show()









