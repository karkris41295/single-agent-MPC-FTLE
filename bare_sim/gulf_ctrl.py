# Running MPC on gulf of mexico
import mpctools as mpc
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

#%%

recall = np.load('/Users/kartikkrishna/Downloads/bigdata/hycom/1992/gulf92.npz')
data = [recall[key] for key in recall]
lon,lat,t_92,uvec,vvec = data

#%%
t_days = np.arange(0,len(t_92),1)
uvec = uvec * 86.4; vvec = vvec*86.4 # kilometers per day
uvec = ma.masked_greater(abs(uvec),1e4)
vvec = ma.masked_greater(abs(vvec),1e4)

x02 = lon * 111 # converting to km
y02 = lat * 111
#%%
# from matplotlib import animation

# fig, ax = plt.subplots(1,1)
# #qui = ax.quiver(x02, y02, uvec[0], vvec[0], color = 'grey')
# ax.set_xlim(lon[0],lon[-1])
# ax.set_ylim(lat[0],lat[-1])

# def update(num,Q):
    
#     ax.set_title('time = ' + str(num))
#     #Q.set_UVC(uvec[num], vvec[num])
#     ax.contourf(lon,lat,(uvec[num]**2 + vvec[num]**2)**.25, cmap = 'winter')
    
#     return Q,

# anim = animation.FuncAnimation(fig, update, fargs=(qui,),interval=1, blit=False, repeat_delay = 10)


#%%

# from scipy.interpolate import RegularGridInterpolator

# ufield = RegularGridInterpolator((t_days, y02, x02), uvec, bounds_error = False, fill_value = 0)
# vfield = RegularGridInterpolator((t_days, y02, x02), vvec, bounds_error = False, fill_value = 0)

#%%

import casadi 

ufieldc = casadi.interpolant('oceanu','linear',[t_days, y02, x02],uvec.ravel(order='F'))
vfieldc = casadi.interpolant('oceanv','linear',[t_days, y02, x02],vvec.ravel(order='F'))

#%%
x1 = casadi.MX.sym('x1')
x2 = casadi.MX.sym('x2')
x3 = casadi.MX.sym('x3')

u1 = casadi.MX.sym('u1')
u2 = casadi.MX.sym('u2')

x = casadi.vertcat(x1,x2,x3)
u = casadi.vertcat(u1,u2)

# %% Define model and parameters.
Delta = .1
ubotmax = 50
Nt = 5 # it was 15
Nx = 3
Nu = 2

# Define stage cost and terminal weight.
Q1 = 1
Q = np.eye(Nx)*Q1
Q[0,0] = 0
Q2 = 0
R1 = 1
R = R1*np.eye(Nu)

#%%

o_field = casadi.vertcat(1, vfieldc(x) + u[1], ufieldc(x) + u[0])
ode_casadi = casadi.Function('F', [x,u], [o_field], ['x', 'u'], ['F'])

k1 = ode_casadi(x, u)
k2 = ode_casadi(x + Delta/2*k1, u)
k3 = ode_casadi(x + Delta/2*k2, u)
k4 = ode_casadi(x + Delta*k3,u) 
xrk4 = x + Delta/6*(k1 + 2*k2 + 2*k3 + k4)
                    
ode_rk4_casadi = casadi.Function("ode_rk4", [x,u], [xrk4])

#%%
goal = np.array([0, 2100, -9300])

def lfunc(x,u):
    """Standard quadratic stage cost."""
    return mpc.mtimes(u.T, R, u) + mpc.mtimes((x-goal).T, Q1, (x-goal))

l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")


def Pffunc(x):
    return Q2*mpc.mtimes((x-goal).T,(x-goal))

Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf") 
l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")

# Bounds on u. Here, they are all [-1, 1]
lb = {"u" : -ubotmax*np.ones((Nu,))}
ub = {"u" : ubotmax*np.ones((Nu,))}

# Make optimizers.
xi = (np.array([0, 2200, -9500]))
N = {"x":Nx, "u":Nu, "t":Nt}

solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, Pf = Pf, x0=xi, lb=lb, ub=ub,verbosity=0, casaditype="MX")
#%%
Nsim = 1000
times = Delta*Nsim*np.linspace(0,1,Nsim+1)

pred = []
upred = []
# Fix initial state.
#solver.fixvar("x", 0, x[0,:])  
for t in range(Nsim):
    #solver.fixvar("x", 0, x[t,:]) 
    # Solve nlp.
    solver.solve()   
    
    # Print stats.
    print("%d: %s" % (t,solver.stats["status"]))
    solver.saveguess()
    solver.fixvar("x",0,solver.var["x",1])
    
    # calling solver variables. Can pull out predicted trajectories from here.
    pred += [solver.var["x",:,:]]
    upred += [solver.var["u",:,:]]
    
    # Simulate.
    # x[t+1,:] = ocean.sim(x[t,:],u[t,:])
    
#%%
# Retrieving trajectory predictions
pred2 = []
pred3 = []    
for i in pred:
    pred2 = []
    for j in i:
        temp = np.array(j)
        pred2 += [temp]
    pred3 += [pred2]

pred3 = np.array(pred3)[:,:,:,0]

# Retrieving energy predictions
upred2 = []
upred3 = []    
for i in upred:
    upred2 = []
    for j in i:
        temp = np.array(j)
        upred2 += [temp]
    upred3 += [upred2]

upred3 = np.array(upred3)[:,:,:,0]

# %% Load LCS Data
recall = np.load('/Users/kartikkrishna/Downloads/bigdata/hycom/ftle_th40_gulf_zoom1.npz')
data = [recall[key] for key in recall]
x01, y01, solfor, solbac = data
# %% Animation/Plotting

from matplotlib import animation
from matplotlib import cm

x_tr = pred3[:,0,:]
u_tr = upred3[:,0,:]
e_spent = (((u_tr[:,0]**2 + u_tr[:,1]**2))**.5)

fig = plt.figure()
ax = plt.subplot(111, title = 'flow field', ylim = (y01[0,0]/111, y01[-1,-1]/111), xlim = (x01[0,0]/111, x01[-1,-1]/111))
#ax2 = plt.subplot(212,xlim = (0, max(x[:,2])), ylim = (min(e_spent), max(e_spent)), title = 'energy spent')

particle, = ax.plot([], [], marker='o', linewidth = 0, color='black')
#traj, = ax.plot([],[], color='purple', alpha=0.6)

#particle2, = ax2.plot([], [], marker='.', linewidth = 0, color='black')
#traj2, = ax2.plot([],[], color='red', alpha=0.6)

#fortraj, = ax.plot([],[], color='purple', alpha=1, linestyle = '--') # forward trajectory
fortraj, = ax.plot([],[], alpha=1, linestyle = '--')
#ax.scatter(goal[2]/111, goal[1]/111, marker = 'X', color='green', s=90)
plt.tight_layout()

#e_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# for i in range(len(pred3[:,0,0])):
#     ax.plot(pred3[i,:2,2]/111, pred3[i,:2,1]/111, alpha=0.3, c = cm.plasma(e_spent[i]/max(1.00001*e_spent)), linewidth = 3)

def update(num):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    ax.collections = []
    ax.contour(x01/111, y01/111, solfor[num], origin = 'lower', cmap = 'winter', alpha = 1)
    ax.contour(x01/111, y01/111, solbac[num], origin = 'lower', cmap = 'autumn', alpha = 1)
    
    num = num*int(1/Delta)
    particle.set_data(x_tr[num,2]/111, x_tr[num,1]/111)
    ax.set_title('e_spent =' + str(e_spent[num]))
    #e_text.set_text('energy fraction relative to max capacity = %.5f' % e_spent[num])
    #traj.set_data(x[:num+1, 0],x[:num+1, 1])
    fortraj.set_data(pred3[num,:,2]/111, pred3[num,:,1]/111)
    fortraj.set_color(cm.plasma(2*e_spent[num]/max(1.000001*e_spent)))

    ax.scatter(goal[2]/111, goal[1]/111, marker = 'X', color='green', s=180)
    ax.scatter(xi[2]/111, xi[1]/111, marker = 'X', color='red', s=180)
    
    #particle2.set_data(x[num,2], e_spent[num])
    #traj2.set_data(x[:num+1,2], e_spent[:num+1])
    
    return particle, fortraj

anim = animation.FuncAnimation(fig, update, blit=False, interval=1000)

#%%

plt.contourf(lon,lat,(uvec[1]**2 + vvec[1]**2)**.25, cmap = 'winter')

xtic = np.linspace(lon[10], lon[-1], 5)
ytic = np.round(np.linspace(lat[10], lat[-1], 5), 1)
plt.xticks(xtic,fontsize = 18)
plt.yticks(ytic,fontsize = 18)
plt.tight_layout()

#%%
x_tr = pred3[:,0,:]
u_tr = upred3[:,0,:]
e_spent = (((u_tr[:,0]**2 + u_tr[:,1]**2))**.5)

fig = plt.figure()
ax = plt.subplot(111, title = 'flow field', ylim = (y01[0,0]/111, y01[-1,-1]/111), xlim = (x01[0,0]/111, x01[-1,-1]/111))
#ax2 = plt.subplot(212,xlim = (0, max(x[:,2])), ylim = (min(e_spent), max(e_spent)), title = 'energy spent')

particle, = ax.plot([], [], marker='o', linewidth = 0, color='black', markersize = 10)
#traj, = ax.plot([],[], color='purple', alpha=0.6)

#particle2, = ax2.plot([], [], marker='.', linewidth = 0, color='black')
#traj2, = ax2.plot([],[], color='red', alpha=0.6)

#fortraj, = ax.plot([],[], color='purple', alpha=1, linestyle = '--') # forward trajectory
fortraj, = ax.plot([],[], alpha=1, linestyle = '--')
#ax.scatter(goal[2]/111, goal[1]/111, marker = 'X', color='green', s=199)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.tight_layout()

#e_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


for i in range(len(pred3[:,0,0])):
    ax.plot(pred3[i,:2,2]/111, pred3[i,:2,1]/111, alpha=0.3, c = cm.plasma(e_spent[i]/max(1.00001*e_spent)), linewidth = 3)

update(11)
#%%


plt.subplot(111)
plt.plot(times[:-1],e_spent,'black')
plt.scatter(np.array([6,7,8,10,12,13]), [e_spent[6*int(1/Delta)], e_spent[7*int(1/Delta)], e_spent[8*int(1/Delta)], e_spent[10*int(1/Delta)], e_spent[12*int(1/Delta)], e_spent[13*int(1/Delta)]], c= 'black', s= 90)
plt.xlim((5,14))
plt.ylim(22, 37)
plt.yticks(fontsize = 24)
plt.xticks(fontsize = 24)
plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=10)
plt.tight_layout()
