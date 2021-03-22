# Code to read double sweep data
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

with open('swp1t11sp5natoq1guess', 'rb') as f:  
    A, eps, omega, Delta, Nsim, Nt, Nu, Nx, d_sweep_data = pickle.load(f)

colors = cm.nipy_spectral(np.linspace(0, 1, len(d_sweep_data)))

plt.figure(figsize = (10,10))
i = 0
for ths in d_sweep_data:
    th = ths[0]
    rq = np.array(ths[6])
    es2 = np.array(ths[2])
    uru = rq*es2
    tc = np.array(ths[5])
    plt.scatter((tc-uru)*Delta,es2*Delta, marker = '.', label = '$T_H$ = '+str('%.1f'%(th*Delta)), c = colors[i])
    i+=1

plt.legend()
plt.xlabel('$\sum{(x-goal)^T(x-goal)} \Delta t$')
plt.ylabel('$\sum{u^Tu} \Delta t$')
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
# %%
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

fname = 'swp1t11sp5natoq1guess'
with open(fname, 'rb') as f:  
    A, eps, omega, Delta, Nsim, Nt, Nu, Nx, d_sweep_data = pickle.load(f)

colors = cm.nipy_spectral(np.linspace(0, 1, len(d_sweep_data)+1))

# slicing the array
slc_start = 0
slc_end = -1

i = 0
for ths in d_sweep_data:
    th = ths[0]
    rq = np.array(ths[6])
    es2 = np.array(ths[2])
    uru = rq*es2
    tc = np.array(ths[5])
    x = (tc-uru)*Delta
    y = es2*Delta
    z = th*Delta
    #ax.scatter((tc-uru)*Delta,es2*Delta,th*Delta, marker = '.', c = 'grey', alpha = .05)
    #ax.scatter(x[slc_start:slc_end],y[slc_start:slc_end],z,label = '$T_H$ = '+str(z), marker = '.', c = colors[i], alpha=1)
    #ax.scatter(x[slc_start:slc_end],y[slc_start:slc_end],z, marker = '.', c = colors[i])
    if i%2 != 0:
        ax.scatter(x[slc_start:slc_end],y[slc_start:slc_end], marker = 'o', c= colors[i-1], alpha = 1)
        
        #ax.scatter(x[slc_start:slc_end],y[slc_start:slc_end],z,label = '$T_H$ = '+str(z), marker = '.', c = colors[i], alpha=1)
        print(th)
        print(i)
    i+=1

if fname == 'swp1t11sp5natoq1guess':
    i = 0
    for ths in d_sweep_data:
        th = ths[0]
        rq = np.array(ths[6])
        es2 = np.array(ths[2])
        uru = rq*es2
        tc = np.array(ths[5])
        x = (tc-uru)*Delta
        y = es2*Delta
        z = th*Delta
        #ax.scatter((tc-uru)*Delta,es2*Delta,th*Delta, marker = '.', c = 'grey', alpha = .05)
        #ax.scatter(x[slc_start:slc_end],y[slc_start:slc_end],z,label = '$T_H$ = '+str(z), marker = '.', c = colors[i], alpha=1)
        #ax.scatter(x[slc_start:slc_end],y[slc_start:slc_end],z, marker = '.', c = colors[i])
        if i == 3:
            k = 79
            ax.scatter(x[[57,k,97]],y[[57,k,97]], marker = 'o', c= 'black', s = 400)
            ax.scatter(x[[57,k,97]],y[[57,k,97]], marker = 'o', c= 'white', s = 300)
            ax.scatter(x[[57,k,97]],y[[57,k,97]], marker = 'o', c= 'black', s = 150)
            ax.scatter(x[[57,k,97]],y[[57,k,97]], marker = 'o', c= colors[i-1], s = 100)
            
            print(th)
            print('E = ' + str(x[[57,k,97]]) + ' U = ' + str(y[[57,k,97]]))
        i+=1


# if slc_end == -1: slc_end = len(x)
# ax.text2D(0.05, 0.95, "$R/Q$ range: "+str(10**(slc_start*.04 -2))+" - "+str(10**(slc_end*.04 -2)), transform=ax.transAxes)
# ax.set_xlabel('$\sum {e}^T{e}$')
# ax.set_ylabel('$\sum {u}^T{u}$')
# ax.set_zlabel('$T_H$')

plt.xlim((0,90))
plt.ylim((0,9.5))
plt.xticks([0, 20, 40, 60, 80],fontsize=30)
plt.yticks([2, 4, 6, 8],fontsize=30)
plt.tight_layout()
#ax.legend()

#%% 

# np.random.seed(19680)

# plt.subplot(111)
# plt.imshow(np.random.random((100, 100))*10 + 1, cmap=plt.cm.nipy_spectral)

# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.colorbar(cax=cax)
# plt.yticks(fontsize=30)

# %% Visualize a subset of data

# with open('swp1t11sp5natoq1om0p7guess', 'rb') as f:  
#     A, eps, omega, Delta, Nsim, Nt, Nu, Nx, d_sweep_data = pickle.load(f)
    
# ths = d_sweep_data[14]
# th = ths[0]
# rq = np.array(ths[6])
# es2 = np.array(ths[2])
# uu = es2
# uru = es2*rq
# tc = np.array(ths[5])

# fig = plt.figure()
# ax2 = plt.subplot(111, xlabel = 'R/Q', xlim=(0, max(rq)), title = '$T_H$ = ' + str(Delta*th))
# plt.tight_layout()
# #qr_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# '''
# ax2.plot(rq,uu*Delta, 'blue', alpha = .3, label = '$\sum (u^Tu) \Delta t$')
# ax2.plot(rq, (tc-uru)*Delta, 'red', alpha = .3, label = '$\sum (x^Tx) \Delta t$')
# ax2.plot(rq,((tc-uru) + uu)*Delta, 'purple', alpha = .3, label = '$\sum (x^Tx + u^Tu) \Delta t$')
# '''
# ax2.plot(rq,uru*Delta, 'blue', alpha = .3, label = '$\sum (u^TRu) \Delta t$')
# ax2.plot(rq,(tc-uru)*Delta, 'red', alpha = .3, label = '$\sum (x^TQx) \Delta t$')
# ax2.plot(rq,tc*Delta, 'purple', alpha = .3, label = '$\sum (x^TQx + u^TRu) \Delta t$')
# ax2.set_xscale('symlog')

# ax2.legend()

#%% Plotting pareto curve against gyre frequency

# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111, projection='3d')

# sepsims = ['swp1t11sp5natoq1om2guess', 'swp1t11sp5natoq1om4guess', 'swp1t11sp5natoq1om6guess', 'swp1t11sp5natoq1om8guess', 'swp1t11sp5natoq1guess', 'swp1t11sp5natoq1om12guess', 'swp1t11sp5natoq1om14guess', 'swp1t11sp5natoq1om16guess', 'swp1t11sp5natoq1om18guess', 'swp1t11sp5natoq1om20guess']
# #sepsims = ['swp1t11sp5natoq1om3p1guess']

# colors = cm.nipy_spectral(np.linspace(0, 1, len(sepsims)+1))

# # slicing the array
# slc_start = 0
# slc_end = -1

# i = 0
# for sim in sepsims:
    
#     with open(sim, 'rb') as f:  
#         A, eps, omega, Delta, Nsim, Nt, Nu, Nx, d_sweep_data = pickle.load(f)
    
#     ths = d_sweep_data[14] # Double gyre time
#     th = ths[0]
#     rq = np.array(ths[6])
#     es2 = np.array(ths[2])
#     uru = rq*es2
#     tc = np.array(ths[5])
#     x = (tc-uru)*Delta
#     y = es2*Delta
#     z = omega
#     #ax.scatter((tc-uru)*Delta,es2*Delta,th*Delta, marker = '.', c = 'grey', alpha = .05)
#     ax3.scatter(x[slc_start:slc_end],y[slc_start:slc_end],2*np.pi/z, marker = '.', c = colors[i], alpha=1)
#     #ax.scatter(x[slc_start:slc_end],y[slc_start:slc_end],z, marker = '.', c = colors[i])
#     i+=1

# if slc_end == -1: slc_end = len(x)
# ax3.text2D(0.05, 0.95, "$R/Q$ range: "+str(10**(slc_start*.04 -2))+" - "+str(10**(slc_end*.04 -2)) + ", $T_H = $ " + str(Delta*th), transform=ax3.transAxes)
# ax3.set_xlabel('$\sum {e}^T{e}$')
# ax3.set_ylabel('$\sum {u}^T{u}$')
# ax3.set_zlabel('$T_{DG}$')
# #ax.legend()
