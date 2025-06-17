from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from pyeit.mesh.meshpy.build import create

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines

import pyeit.eit.jac as jac
from pyeit.eit.interp2d import sim2pts
import time
import pandas as pd


start_time = time.time()


def circle(num_poly):
    data = np.genfromtxt("Data/circle.csv", dtype=float, delimiter=',', names=True) 
    circle_point = []
    for i in range(32):
        circle_point.append([data['x'][i]/100,data['y'][i]/100])
    
    pts = circle_point
    pts = pts
    n = [np.size(pts, 0)]
    return pts, n

N_el = 8
mesh_obj, el_pos = create(N_el, max_area=0.00005, curve=circle, refine=True)
pts = mesh_obj['node']
tri = mesh_obj['element']
x, y = pts[:, 0], pts[:, 1]


#number_element = np.shape(tri)[0]
#for i in range(number_element):
#    cx = (x[mesh_obj['element'][i][0]]+x[mesh_obj['element'][i][1]]+x[mesh_obj['element'][i][2]])/3
#    cy = (y[mesh_obj['element'][i][0]]+y[mesh_obj['element'][i][1]]+y[mesh_obj['element'][i][2]])/3
#    point = Point(cx, cy)
#    if (polygon.contains(point)):
#        mesh_obj['perm'][i]=20
    
perm = mesh_obj['perm']

el_pos = []
for elp in range (N_el):
    el_pos.append(4*elp)

# el_pos = [ 0,4,8,12,16,20,24,28 ]
# el_pos = [ 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 ]
# el_pos = [ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]


el_pos = np.array(el_pos)
fig, ax = plt.subplots(figsize=(5, 5))
plt.triplot(pts[:, 0], pts[:, 1], tri)

# fig, ax = plt.subplots(figsize=(5, 5))


ax.tripcolor(x, y, tri, np.real(perm),
              edgecolors='k', shading='flat', alpha=0.5,
              cmap=plt.cm.Greys)

for i, e in enumerate(el_pos):
    plt.plot(pts[el_pos[i], 0], pts[el_pos[i], 1], 'ro')
    plt.annotate(str(i+1), xy=(x[e], y[e]), color='k')
    
#splt.axis([-0.5,0.5,-0.5,0.5])
plt.grid()
plt.show()


""" 2. FEM simulation """
el_dist, step = 1, 1
ex_mat = eit_scan_lines(N_el, el_dist)


""" 3. JAC solver """
eit = jac.JAC(mesh_obj, el_pos, ex_mat=ex_mat, step=step,
              perm=1., parser='std', jac_normalized=False)

""" 4. Initial setup """
stomachRecord = []
signalRecord = []
measNum = []
initialCount = 0
Current = 1#mA
frequency = 10000#mA



allDSN = []
t = 1
for m in range(t):

    directory = 'Data/'
    # bgfile = 'Background_Current=1mA_stomach_fasting_f=10000.0Hz'
    # objfile = 'Data_'+str(m+1)+'_Current=1mA_stomach_full_f=10000.0Hz'
    
    bgfile = 'Homogen'
    objfile = 'Inclusion'

    data2 = pd.read_csv(directory+bgfile+'.csv')
    data1 = pd.read_csv(directory+objfile+'.csv')
    
    Z_Inclusion = np.array(data1['Z [Ohm]'])
    Z_mean = np.ones(40)
    Z_mean = Z_mean*np.mean(Z_Inclusion)
    Z_Homogen = np.array(data2['Z [Ohm]'])
    
    
    Zh = list(Z_Homogen)
    Zi = list(Z_Inclusion)
    
    vi = Z_Inclusion
    vh = Z_Homogen
    d_error = []
    c_error = []
    
    pVal=0.4
    lambVal=0.01
    eit.setup(p=pVal, lamb=lambVal, method='kotre')
    # ds = eit.solve_gs(vi, vh)
    ds = eit.solve(vi,vh,normalize=True)
    # ds = eit.jt_solve(vi, vh, normalize=False)
    ds[ds<0]=0
    ds[ds>10]=10
    
    # ds = eit.gn((vi-vh)/10, lamb_decay=0.01, lamb_min=1e-4, maxiter=1, verbose=True)
    ds_n = sim2pts(pts, tri, np.real(ds))
    
    print(np.shape(x),np.shape(y),np.shape(tri),np.shape(ds_n))
    
    allDSN.append(ds_n)
    
    
m = 0
for m in range(t):
    
    # plot EIT reconstruction
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.tripcolor(x, y, tri, allDSN[m], shading='flat',cmap=plt.cm.jet)
    # for i, e in enumerate(el_pos):
    #     ax.annotate(str(i+1), xy=(x[e], y[e]), color='r')
    # fig.colorbar(im)
    ax.axis('off')
    # ax.set_aspect('equal')
    # fig.set_size_inches(6, 4)
    plt.savefig(directory+'Img_Adj_'+objfile+str(m)+'_p'+str(pVal)+'_lamb'+str(lambVal)+'.png', dpi=96)
    
    plt.show()
    print(np.mean(np.real(ds)))
    print("measTime=", time.time()-start_time)
    
    "plot empying by signal 16 1 2 3"
    # observed_signal = 1/(np.array(Zi[(13*15):13*16]+Zi[0:13*3]))
    observed_signal = 1/(vi)
    signalRecord.append(np.mean(observed_signal))
    measNum.append(m)
    
    if (m==0):
        initialSignal = sum(signalRecord)