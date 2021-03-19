#Plotting results and running MPC for Double Gyre
import mpctools as mpc
import matplotlib.pyplot as plt
import numpy as np

# %% Load LCS Data
recall = np.load('ftle_th15.npz')
data = [recall[key] for key in recall]
x0, y0, solfor, solbac = data

# %% Paramteters from Shadden Physica D
A = .1
eps = .25
omega = 2*np.pi/10

# %% Define model and parameters.
Delta = .1
ubotmax = .1
Nt = 70 # it was 15
Nx = 3
Nu = 2

# Define stage cost and terminal weight.
Q1 = 1
Q = np.eye(Nx)*Q1
Q[2,2] = 0
Q2 = 0
R1 = 51
R = R1*np.eye(Nu)

def dgyre(x, u, A = A, eps = eps, om = omega):
    """Continuous-time ODE model."""
    
    a = eps * np.sin(om * x[2]);
    b = 1 - 2 * a;
    
    f = a * x[0]**2 + b * x[0];
    df = 2 * a * x[0] + b;
    
    dxdt = [
        -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * x[1]) + u[0],
        np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * x[1]) * df + u[1],
        1,
    ]
    return np.array(dxdt)

# Create a simulator. This allows us to simulate a nonlinear plant.
ocean = mpc.DiscreteSimulator(dgyre, Delta, [Nx,Nu], ["x","u"])

# Then get casadi function for rk4 discretization.
ode_rk4_casadi = mpc.getCasadiFunc(dgyre, [Nx,Nu], ["x","u"], funcname="F",
    rk4=True, Delta=Delta, M=1)

goal = np.array([0.5, 0.5, 0])

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
xi = np.array([2,1,0])
N = {"x":Nx, "u":Nu, "t":Nt}
solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, Pf = Pf, x0=xi, lb=lb, ub=ub,verbosity=0)
#solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, x0=xi, lb=lb, ub=ub,verbosity=0)
# %% Now simulate.
Nsim = 800
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
    solver.saveguess()
    solver.fixvar("x",0,solver.var["x",1])
    
    u[t,:] = np.array(solver.var["u",0,:]).flatten() 
    # calling solver variables. Can pull out predicted trajectories from here.
    pred += [solver.var["x",:,:]]
    upred += [solver.var["u",:,:]]
    
    # Simulate.
    x[t+1,:] = ocean.sim(x[t,:],u[t,:])

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

# %% Animation/Plotting

# from matplotlib import animation
# from matplotlib import cm

# e_spent = (((u[:,0]**2 + u[:,1]**2))**.5)

# fig = plt.figure()
# plt.style.use('dark_background')
# ax = plt.subplot(211, aspect = 'equal', xlim=(0, 2), ylim=(0, 1), title = 'flow field')
# ax2 = plt.subplot(212,xlim = (0, max(x[:,2])), ylim = (min(e_spent), max(e_spent)), title = 'energy spent')

# particle, = ax.plot([], [], marker='o', linewidth = 0, color='white')
# traj, = ax.plot([],[], color='purple', alpha=0.6)

# particle2, = ax2.plot([], [], marker='.', linewidth = 0, color='black')
# traj2, = ax2.plot([],[], color='red', alpha=0.6)

# #fortraj, = ax.plot([],[], color='purple', alpha=1, linestyle = '--') # forward trajectory
# fortraj, = ax.plot([],[], alpha=1, linestyle = '--')
# ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)
# plt.tight_layout()

# #e_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# # for i in range(Nsim):
# #     ax.plot(pred3[i,:,0], pred3[i,:,1], alpha=0.01, c = 'grey')

# def update(num):
#     """updates the horizontal and vertical vector components by a
#     fixed increment on each frame
#     """
#     ax.collections = []
#     ax.contour(x0, y0, solfor[num], origin = 'lower', cmap = 'winter', alpha = 1)
#     ax.contour(x0, y0, solbac[num], origin = 'lower', cmap = 'autumn', alpha = 1)
    
#     num = num*int(.2/Delta)
#     particle.set_data(x[num,0], x[num,1])
#     #e_text.set_text('energy fraction relative to max capacity = %.5f' % e_spent[num])
#     traj.set_data(x[:num+1, 0],x[:num+1, 1])
#     # fortraj.set_data(pred3[num,:,0], pred3[num,:,1])
#     # fortraj.set_color(cm.plasma(2*e_spent[num]/max(1.000001*e_spent)))

#     ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)
    
#     particle2.set_data(x[num,2], e_spent[num])
#     traj2.set_data(x[:num+1,2], e_spent[:num+1])
    
#     return particle, traj, fortraj

# anim = animation.FuncAnimation(fig, update, blit=False, interval=100)

#%% Plot with vector field

from matplotlib import cm
from matplotlib import animation

e_spent = (((u[:,0]**2 + u[:,1]**2))**.5)
plt.style.use('dark_background')
fig = plt.figure(figsize = (10, 5))

ax = plt.subplot(121, aspect = 'equal', xlim=(-0, 2), ylim=(-0, 1), title = 'MPC Trajectory with $T_H = $' + str(Nt*Delta) + ', $R/Q = $' + str(R1) + '\n $\Delta t = $ ' + str(Delta) + ', Simulation runtime = ' + str(Nsim*Delta))
ax2 = plt.subplot(222, title = 'costs along dashed forward horizon', xlabel = '$t$', xlim=(0, max(times)))
ax3 = plt.subplot(224, title = 'instant energy spent', xlabel = '$t$', xlim=(0, max(times)))

# =============================================================================
# for i in range(num+20):
#     ax.plot(pred3[i,:,0], pred3[i,:,1], alpha=0.2, c = cm.plasma(e_spent[i]/max(1.00001*e_spent)))
# =============================================================================
    
en_segment = R1*np.sum((upred3[:,:,0]**2 + upred3[:,:,1]**2)**.5, axis=1)
er_segment = np.sum(((upred3[:,:,0]-goal[0])**2 + (upred3[:,:,1]-goal[1])**2)**.5, axis=1)
ax2.plot(times[:-1], en_segment, 'grey', alpha = .3)
ax2.plot(times[:-1], er_segment, 'grey', alpha = .3)
ax2.plot(times[:-1], en_segment + er_segment, 'grey', alpha = .3)
ax3.plot(times[:-1], e_spent, 'grey', alpha = .3)
particle4, = ax3.plot([], [], marker='o', linewidth = 0, color='black')
particle1, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
particle2, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
particle3, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
traj1, = ax2.plot([],[], color='blue', alpha=0.6, label = '$\sum (u^TRu)$')
traj2, = ax2.plot([],[], color='red', alpha=0.6, label = '$\sum (x^TQx)$')
traj3, = ax2.plot([],[], color='purple', alpha=0.6, label = '$\sum (x^TQx + u^TRu)$')
traj4, = ax3.plot([],[], color='green', alpha=0.6)
#ax2.legend()

#traj, = ax.plot([],[], color='cyan', alpha=0.6)

particle, = ax.plot([], [], marker='o', linewidth = 0, color='white', markersize = '8')

#fortraj, = ax.plot([],[], color='cyan', alpha=1, linestyle = '--') # forward trajectory
fortraj, = ax.plot([],[], alpha=1, linestyle = '--')
plt.tight_layout()

# def doublegyreVEC(t, yin, A, eps, om):
#     x = yin[0]
#     y = yin[1]

#     u = np.zeros(x.shape); 
#     v = u.copy()

#     a = eps * np.sin(om * t);
#     b = 1 - 2 * a;
    
#     f = a * x**2 + b * x;
#     df = 2 * a * x + b;

#     u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y);
#     v =  np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * df;

#     return np.array([u,v])

# dx = .1
# xvec = np.arange(0-dx, 2+dx, dx)
# yvec = np.arange(0-dx, 1+dx, dx)

# x0, y0 = np.meshgrid(xvec, yvec)
# yIC = np.zeros((2, len(yvec), len(xvec)))
# yIC[0], yIC[1] = x0, y0


# dy = doublegyreVEC(0,yIC,A,eps,omega)
#qui = ax.quiver(x0, y0, dy[0], dy[1], color = 'grey')

def update(num,Q):
    #t_snap = Delta*num
    ax.collections = []
    ax.contour(x0, y0, solfor[num], origin = 'lower', cmap = 'winter', alpha = 1)
    ax.contour(x0, y0, solbac[num], origin = 'lower', cmap = 'autumn', alpha = 1)

    # dy = doublegyreVEC(t_snap,yIC,A,eps,omega)
    # Q.set_UVC(dy[0], dy[1])
    
    num = num*2
    particle1.set_data(times[num], en_segment[num])
    traj1.set_data(times[:num+1],en_segment[:num+1]) 
    particle2.set_data(times[num], er_segment[num])
    traj2.set_data(times[:num+1],er_segment[:num+1]) 
    particle3.set_data(times[num], en_segment[num]+er_segment[num])
    traj3.set_data(times[:num+1],en_segment[:num+1]+er_segment[:num+1]) 
    
    particle4.set_data(times[num], e_spent[num])
    traj4.set_data(times[:num+1],e_spent[:num+1]) 
    
    fortraj.set_data(pred3[num,:,0], pred3[num,:,1])
    #fortraj.set_color(cm.plasma(2*e_spent[num]/max(1.000001*e_spent)))
    fortraj.set_color(cm.plasma(e_spent[num]/.04))
    #ax.plot(pred3[num,:,0], pred3[num,:,1], color=cm.plasma(1.4*e_spent[num]/max(1.000001*e_spent)), alpha=.1)
    
    ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)
    particle.set_data(x[num,0], x[num,1])
    
    #traj.set_data(x[:num+1, 0],x[:num+1, 1])
    
#anim = animation.FuncAnimation(fig, update, fargs=(qui,),interval=100, blit=False, repeat_delay = 10)
    
anim = animation.FuncAnimation(fig, update, fargs=(1,),interval=100, blit=False, repeat_delay = 10)
# %% Plotting color energy
'''
from matplotlib import animation

e_spent = (((u[:,0]**2 + u[:,1]**2))**.5)

fig = plt.figure(figsize = (10, 5))
ax = plt.subplot(121, aspect = 'equal',title ='MPC Trajectory with $T_H = $' + str(Nt*Delta) + ', $R/Q1 = $' + str(R1/Q1) +', $Q2 = $' + str(Q2) + '\n $\Delta t = $ ' + str(Delta) + ', Simulation runtime = ' + str(Nsim*Delta)+ ', $\omega = $ ' + str('%.3f'%(omega)), xlim=(-.1, 2.1), ylim=(-.1, 1.1))
ax2 = plt.subplot(222, title = 'costs along dashed forward horizon', xlabel = '$t$', xlim=(0, max(times)))
ax3 = plt.subplot(224, title = 'instant energy cost', xlabel = '$t$', xlim=(0, max(times)))

en_segment = R1*np.sum((upred3[:,:,0]**2 + upred3[:,:,1]**2)**.5, axis=1)
er_segment = np.sum(((upred3[:,:,0]-goal[0])**2 + (upred3[:,:,1]-goal[1])**2)**.5, axis=1)
ax2.plot(times[:-1], en_segment, 'grey', alpha = .3)
ax2.plot(times[:-1], er_segment, 'grey', alpha = .3)
ax2.plot(times[:-1], en_segment + er_segment, 'grey', alpha = .3)
ax3.plot(times[:-1], e_spent, 'grey', alpha = .3)
particle4, = ax3.plot([], [], marker='o', linewidth = 0, color='black')
particle1, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
particle2, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
particle3, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
traj1, = ax2.plot([],[], color='blue', alpha=0.6, label = '$\sum (u^TRu)$')
traj2, = ax2.plot([],[], color='red', alpha=0.6, label = '$\sum (x^TQx)$')
traj3, = ax2.plot([],[], color='purple', alpha=0.6, label = '$\sum (x^TQx + u^TRu)$')
traj4, = ax3.plot([],[], color='green', alpha=0.6)
#ax2.legend()

particle, = ax.plot([], [], marker='o', linewidth = 0, color='black')
traj, = ax.plot([],[], color='cyan', alpha=0.6)

#fortraj, = ax.plot([],[], color='cyan', alpha=1, linestyle = '--') # forward trajectory
plt.tight_layout()

from matplotlib import cm

for i in range(Nsim):
    ax.plot(pred3[i,:,0], pred3[i,:,1], alpha=0.06, c = cm.plasma(e_spent[i]/max(1.00001*e_spent)))



def update(num):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    ax.collections = []
    ax.contour(x0, y0, solfor[num], origin = 'lower', cmap = 'winter', alpha = 1)
    ax.contour(x0, y0, solbac[num], origin = 'lower', cmap = 'autumn', alpha = 1)
    
    num = num*int(.2/Delta)
    particle.set_data(x[num,0], x[num,1])
    ax.quiver(x[num,0], x[num,1], u[num,0], u[num,1], scale_units= 'xy', scale = .3*np.mean(e_spent)/e_spent)
    #e_text.set_text('energy fraction relative to max capacity = %.5f' % e_spent[num])
    #traj.set_data(x[:num+1, 0],x[:num+1, 1])
    fortraj.set_data(pred3[num,:,0], pred3[num,:,1])
    ax.scatter(goal[0], goal[1], color='green')


    particle1.set_data(times[num], en_segment[num])
    traj1.set_data(times[:num+1],en_segment[:num+1]) 
    particle2.set_data(times[num], er_segment[num])
    traj2.set_data(times[:num+1],er_segment[:num+1]) 
    particle3.set_data(times[num], en_segment[num]+er_segment[num])
    traj3.set_data(times[:num+1],en_segment[:num+1]+er_segment[:num+1]) 
    
    particle4.set_data(times[num], e_spent[num])
    traj4.set_data(times[:num+1],e_spent[:num+1]) 
    
    return particle, traj, fortraj

anim = animation.FuncAnimation(fig, update, blit=False, interval=300)
'''
