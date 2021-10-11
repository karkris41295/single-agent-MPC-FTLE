#Plotting results and running MPC for ABC flow
import mpctools as mpc
import matplotlib.pyplot as plt
import numpy as np

# %% Paramteters 
A = 3**.5 * .1
B = 2**.5 * .1
C = 1**.5 * .1
amp = 0.01
om = 2*np.pi/10

# %% Define model and parameters.
Delta = .1
ubotmax = A+B+C
Nt = 30 # it was 15
Nx = 4
Nu = 3

# Define stage cost and terminal weight.
Q1 = 1
Q = np.eye(Nx)*Q1
Q[-1,-1] = 0
Q2 = 0
R1 = 25
R = R1*np.eye(Nu)

def dgyre(x, u, A = A, B=B, C=C):
    """Continuous-time ODE model."""
    
    dxdt = [
        (A+amp*np.cos(om*x[3]))*np.sin(x[2]) + C*np.cos(x[1]) + u[0],
        B*np.sin(x[0]) + (A+amp*np.cos(om*x[3]))*np.cos(x[2]) + u[1],
        B*np.cos(x[0]) + C*np.sin(x[1]) + u[2],
        1,
    ]
    return np.array(dxdt)

# Create a simulator. This allows us to simulate a nonlinear plant.
ocean = mpc.DiscreteSimulator(dgyre, Delta, [Nx,Nu], ["x","u"])

# Then get casadi function for rk4 discretization.
ode_rk4_casadi = mpc.getCasadiFunc(dgyre, [Nx,Nu], ["x","u"], funcname="F",
    rk4=True, Delta=Delta, M=1)

goal = np.array([5, 2, 1, 0])

def lfunc(x,u):
    """Standard quadratic stage cost."""
    return mpc.mtimes(u.T, R, u) + mpc.mtimes((x-goal).T, Q1, (x-goal))

def Pffunc(x):
    return Q2*mpc.mtimes((x-goal).T,(x-goal))

Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf") 

l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")


# Bounds on u. Here, they are all [-1, 1]
lb = {"u" : -ubotmax*np.ones((Nu,))}
ub = {"u" : ubotmax*np.ones((Nu,))}

# Make optimizers.
xi = np.array([np.pi/2,1,6,0])
N = {"x":Nx, "u":Nu, "t":Nt}
solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, Pf = Pf, x0=xi, lb=lb, ub=ub,verbosity=0)
#solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, x0=xi, lb=lb, ub=ub,verbosity=0)
# %% Now simulate.
Nsim = 2000
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = xi
u = np.zeros((Nsim,Nu))
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
    #solver.saveguess()
    solver.fixvar("x",0,solver.var["x",1])
    
    u[t,:] = np.array(solver.var["u",0,:]).flatten() 
    # calling solver variables. Can pull out predicted trajectories from here.
    pred += [solver.var["x",:,:]]
    upred += [solver.var["u",:,:]]
    
    # Simulate.
    #x[t+1,:] = ocean.sim(x[t,:],u[t,:])


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

#%%
x = pred3[:,0,:]

# %% Load LCS Data
recall = np.load('/Users/kartikkrishna/Downloads/bigdata/hycom/ftle_abc_th8hi_dg.npz')
data = [recall[key] for key in recall]
x01, y01, z01, solfor, solbac = data
#%% Plot with vector field (Has capacity for control pointing vector)

from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib import animation

e_spent = (((u[:,0]**2 + u[:,1]**2 + u[:,2]**2))**.5)

fig = plt.figure()
ax = fig.gca(projection='3d')
# =============================================================================
# for i in range(num+20):
#     ax.plot(pred3[i,:,0], pred3[i,:,1], alpha=0.2, c = cm.plasma(e_spent[i]/max(1.00001*e_spent)))
# =============================================================================
    
en_segment = R1*np.sum((upred3[:,:,0]**2 + upred3[:,:,1]**2)**.5, axis=1)
er_segment = np.sum(((upred3[:,:,0]-goal[0])**2 + (upred3[:,:,1]-goal[1])**2)**.5, axis=1)


traj, = ax.plot([],[],[], color='cyan', alpha=0.6)

particle, = ax.plot([],[],[], marker='o', color='black', markersize=8, linewidth = 0)
#particle, = ax.plot([], [], marker='o', linewidth = 0, color='white', markersize = '8')

#fortraj, = ax.plot([],[], color='cyan', alpha=1, linestyle = '--') # forward trajectory
fortraj, = ax.plot([],[], alpha=1, linestyle = '--')
plt.tight_layout()


def doublegyreVEC(t, yin, A, B, C):
    x = yin[0]
    y = yin[1]
    z = yin[2]

    u = np.zeros(x.shape); 
    v = u.copy()
    w = u.copy()

    u = (A+amp*np.sin(om*t))*np.sin(z) + C*np.cos(y);
    v =  B*np.sin(x) + (A+amp*np.sin(om*t))*np.cos(z);
    w = B*np.cos(x) + C*np.sin(y)

    return np.array([u,v,w])

dx = 1
xvec = np.arange(0, 2*np.pi, dx)
yvec = np.arange(0, 2*np.pi, dx)
zvec = np.arange(0, 2*np.pi, dx)

x0, y0, z0 = np.meshgrid(xvec, yvec, zvec)
yIC = np.zeros((3 ,len(zvec), len(yvec), len(xvec)))
yIC[0], yIC[1], yIC[2] = x0, y0, z0

x02, y02 = np.meshgrid(x01[0,:,0], x01[0,:,0])

# dy = doublegyreVEC(0,yIC,A,B,C)
# qui = ax.quiver(x0, y0, z0, dy[0], dy[1], dy[2], color = 'black', length =.25, normalize = True)

def update(num, x, traj, particle):
    #global qui
    t_snap = Delta*num
    
    ax.collections = []
    #ax.contour(x02, y02, solfor[num%50,:,:,53], zdir='z', offset=2.4, cmap='winter')
    #ax.contour(x02, y02, solbac[num%50,:,:,53], zdir='z', offset=2.4, cmap='autumn')
    
    # ax.contour(x02, y02, solfor[num%50,:,:,106], zdir='z', offset=5, cmap='winter')
    # ax.contour(x02, y02, solbac[num%50,:,:,106], zdir='z', offset=5, cmap='autumn')
    
    # ax.contour(x02, y02, solfor[num%50,:,:,85], zdir='z', offset=4, cmap='winter')
    # ax.contour(x02, y02, solbac[num%50,:,:,85], zdir='z', offset=4, cmap='autumn')
    
    # ax.contour(x02, y02, solfor[num%50,:,:,65], zdir='z', offset=3, cmap='winter')
    # ax.contour(x02, y02, solbac[num%50,:,:,65], zdir='z', offset=3, cmap='autumn')
    
    #ax.contour(x02, y02, solfor[num%50,:,:,24], zdir='z', offset=1, cmap='winter')
    #ax.contour(x02, y02, solbac[num%50,:,:,24], zdir='z', offset=1, cmap='autumn')
    
    #ax.contour(x02, y02, solfor[num%50,:,:,0], zdir='z', offset=0, cmap='winter')
    #ax.contour(x02, y02, solbac[num%50,:,:,0], zdir='z', offset=0, cmap='autumn')

    #dy = doublegyreVEC(t_snap,yIC,A,B,C)
    
    num = num*2
    ax.set_title('t = ' + str(t_snap*2))
    ax.set_xlim(0,2*np.pi)
    ax.set_ylim(0,2*np.pi)
    ax.set_zlim(0,2*np.pi)
    #fortraj.set_data(pred3[num,:,0], pred3[num,:,1])
    #fortraj.set_color(cm.plasma(2*e_spent[num]/max(1.000001*e_spent)))
    #fortraj.set_color(cm.plasma(e_spent[num]/.04))
    #ax.plot(pred3[num,:,0], pred3[num,:,1], color=cm.plasma(1.4*e_spent[num]/max(1.000001*e_spent)), alpha=.1)
    #ax.quiver(x[num,0], x[num,1], u[num,0], u[num,1], scale_units= 'xy', scale = .3*np.mean(e_spent)/e_spent, color ='white')
    # qui.remove()
    # qui = ax.quiver(x0, y0, z0, dy[0], dy[1], dy[2], color = 'black', length =.25, normalize = True)
    
    ax.scatter(goal[0], goal[1], goal[2], marker = 'X', color='green', s=90)
    particle.set_data(pred3[:,0,:][num-1:num, :2].T)
    particle.set_3d_properties(pred3[:,0,:][num-1:num, 2].T)
    
    traj.set_data(pred3[:,0,:][:num, :2].T)
    traj.set_3d_properties(pred3[:,0,:][:num, 2].T)
    

anim = animation.FuncAnimation(fig, update,interval=1, blit=False, repeat_delay = 10, fargs=(x, traj, particle))

#%% Plot trajectory

import numpy as np
import matplotlib.pyplot as plt

# Plot
ax = plt.figure().add_subplot(projection='3d')

for i in range(0,len(pred3[:,:,0]),1):
    ax.plot(pred3[i,:2,0], pred3[i,:2,1], pred3[i,:2,2], alpha=0.3, c = cm.plasma(e_spent[i]/max(1.00001*e_spent)), linewidth = 2)
    
ax.set_xlim((0,2*np.pi))
ax.set_ylim((0,2*np.pi))
ax.set_zlim((0,2*np.pi))
ax.scatter(x[0,0],x[0,1],x[0,2], c = 'black', s = 50)
ax.scatter(goal[0], goal[1], goal[2], marker = 'X', color='green', s=90)

ax.contour(x02, y02, solfor[0,:,:,0], zdir='z', offset=0, cmap='winter')
ax.contour(x02, y02, solbac[0,:,:,0], zdir='z', offset=0, cmap='autumn')
    
# print('ax.azim {}'.format(ax.azim))
# print('ax.elev {}'.format(ax.elev))
ax.view_init(7, -50)
plt.show()

