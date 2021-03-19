#Plotting results and running MPC for Double Gyre
import mpctools as mpc
import matplotlib.pyplot as plt
import numpy as np

# # %% Load LCS Data
# recall = np.load('ftle_th15.npz')
# data = [recall[key] for key in recall]
# x0, y0, solfor, solbac = data

# %% Paramteters from Shadden Physica D
A = .1
eps = .25
omega = 2*np.pi/10

# %% Define model and parameters.
Delta = .1
ubotmax = .1
Nt = 40 # it was 15
Nx = 3
Nu = 2

# Define stage cost and terminal weight.
Q1 = 1
Q = np.eye(Nx)*Q1
Q[2,2] = 0
Q2 = 0
R1 = 100
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

goal = np.array([.5, .5, 0])

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
for t in range(Nsim):
    # Fix initial state.
    solver.fixvar("x", 0, x[t,:])  
    
    # Solve nlp.
    solver.solve()   
    
    # Print stats.
    print("%d: %s" % (t,solver.stats["status"]))
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

#%%
def doublegyreVEC(t, yin, A, eps, om):
    x = yin[0]
    y = yin[1]

    u = np.zeros(x.shape); 
    v = u.copy()

    a = eps * np.sin(om * t);
    b = 1 - 2 * a;
    
    f = a * x**2 + b * x;
    df = 2 * a * x + b;

    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y);
    v =  np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * df;

    return np.array([u,v])
    
def rk4singlestep(fun,dt,t0,y0):
    f1 = fun(t0,y0);
    f2 = fun(t0+dt/2,y0+(dt/2)*f1);
    f3 = fun(t0+dt/2,y0+(dt/2)*f2);
    f4 = fun(t0+dt,y0+dt*f3);

    yout = y0 + (dt/6)*(f1+2*f2+2*f3+f4)
    return yout

#%% Part 1 - Initialize grid of particles through vector field

dx = .025
xvec = np.arange(0-dx, 2+dx, dx)
yvec = np.arange(0-dx, 1+dx, dx)

x0, y0 = np.meshgrid(xvec, yvec)
yIC = np.zeros((2, len(yvec), len(xvec)))
yIC[0], yIC[1] = x0, y0


# %% Animate quiver plot

dist_box_x = []
dist_box_y = []

for k in range(1600):
    ttemp = k*.05 # slow down animation
    dy = doublegyreVEC(ttemp,yIC,A,eps,omega)
    dist_box_x+=[dy[0]]
    dist_box_y+=[dy[1]]
    
dist_box_x=np.array(dist_box_x)
dist_box_y=np.array(dist_box_y)


#%%

plt.rcParams.update({'font.size': 30})
# fig = plt.figure()

# plt.hist(dist_box_x[:].flatten(),100, color = 'black', label = 'background flow', alpha =1, weights = np.ones_like(dist_box_x.flatten())/len(dist_box_x.flatten()))
# plt.hist(u[:,0],100, color = 'red', label = 'control action', alpha = .5, weights = np.ones_like(u[:,0])/len(u[:,0]), range = (min(dist_box_x.flatten()),max(dist_box_x.flatten())))

plt.hist(dist_box_x[:].flatten(),100, color = 'black', alpha =1, weights = np.ones_like(dist_box_x.flatten())/len(dist_box_x.flatten()))
plt.hist(u[:,0],100, color = 'red', alpha = .5, weights = np.ones_like(u[:,0])/len(u[:,0]), range = (min(dist_box_x.flatten()),max(dist_box_x.flatten())))
plt.xlim((-.3, .3))
# plt.title('distribution of the x-component velocities')
# plt.ylabel('probability of $u_x$')
#plt.ylim((0,1))
#plt.legend()

plt.yticks([.1,.2])
plt.xticks([-.1,-.3,.1,.3])
plt.ylim((0,.3))
plt.tight_layout()

fig2 = plt.figure()

# plt.hist(dist_box_y[:].flatten(), 100, color = 'black', label = 'background flow', weights = np.ones_like(dist_box_x.flatten())/len(dist_box_x.flatten()))
# plt.hist(u[:,1],100, color = 'red', label = 'control action', alpha=.5, weights = np.ones_like(u[:,1])/len(u[:,0]), range = (min(dist_box_y.flatten()),max(dist_box_y.flatten())))

plt.hist(dist_box_y[:].flatten(), 100, color = 'black', weights = np.ones_like(dist_box_x.flatten())/len(dist_box_x.flatten()))
plt.hist(u[:,1],100, color = 'red', alpha=.5, weights = np.ones_like(u[:,1])/len(u[:,0]), range = (min(dist_box_y.flatten()),max(dist_box_y.flatten())))
# plt.title('distribution of the y-component of velocities')
# plt.ylabel('probability of $u_y$')
#plt.ylim((0,1))
plt.xlim((-.3, .3))
#plt.legend()
plt.yticks([.1,.2])
plt.ylim((0,.3))
plt.xticks([-.1,-.3,.1,.3])
plt.tight_layout()
gyre_energy = (dist_box_y[:].flatten()**2+dist_box_y[:].flatten()**2)**.5
fig3 = plt.figure()
# plt.hist(gyre_energy, 80, color = 'black', label = 'background flow', weights = np.ones_like(dist_box_x.flatten())/len(dist_box_x.flatten()))
# plt.hist((u[:,0]**2+u[:,1]**2)**.5,80, color = 'red', label = 'control action', alpha = .5, weights = np.ones_like(u[:,0])/len(u[:,0]), range = (min(gyre_energy),max(gyre_energy)))

plt.hist(gyre_energy, 80, color = 'black', weights = np.ones_like(dist_box_x.flatten())/len(dist_box_x.flatten()))
plt.hist((u[:,0]**2+u[:,1]**2)**.5,80, color = 'red', alpha = .5, weights = np.ones_like(u[:,0])/len(u[:,0]), range = (min(gyre_energy),max(gyre_energy)))
# plt.title('distribution of the $\||u\||$ of velocities')
# plt.title('distribution of the $\||u\||$ of velocities')
# plt.ylabel('probability of $\||u\||$')
#plt.ylim((0,1))
#plt.yticks([.1,.2])
plt.xticks(np.arange(0.1,.7,.2))
plt.xlim((0, np.pi/10*2**.5))
plt.ylim((0,.42))
# plt.legend()

plt.tight_layout()


#%% finding the background velocity along trajectory
plt.rcParams.update({'font.size': 30})
back_xd = (x[1:,0] - x[:-1,0])/Delta - u[:,0]
back_yd = (x[1:,1] - x[:-1,1])/Delta - u[:,1]

# plt.figure()
# plt.plot(times[:-1], abs(back_xd))
# plt.plot(times[:-1], abs(u[:,0]))
# plt.ylim(-.01,.33)
# plt.xlim(0,times[-1])

# plt.figure()
# plt.plot(times[:-1], abs(back_yd))
# plt.plot(times[:-1], abs(u[:,1]))
# plt.ylim(-.01,.33)
# plt.xlim(0,times[-1])

plt.figure()

# plt.hist(back_xd, 60, color = 'black', label = 'background flow', alpha =1, range = (min(back_xd),max(back_xd)), weights = 1/Nsim* np.ones(Nsim))
# plt.hist(u[:,0],60, color = 'red', label = 'control action', alpha = .5,range = (min(back_xd),max(back_xd)),weights = 1/Nsim* np.ones(Nsim))

plt.hist(back_xd, 60, color = 'grey', alpha =1, range = (min(back_xd),max(back_xd)), weights = 1/Nsim* np.ones(Nsim))
plt.hist(u[:,0],60, color = 'red', alpha = .5,range = (min(back_xd),max(back_xd)),weights = 1/Nsim* np.ones(Nsim))
plt.xlim((-.3, .3))
#plt.title('distribution of the x-component velocities')
#plt.ylabel('probability of $u_x$')
#plt.ylim((0,1))
plt.xlim((-.3, .3))
plt.yticks([.1,.2])
plt.ylim((0,.3))
plt.xticks([-.1,-.3,.1,.3])

plt.tight_layout()

fig2 = plt.figure()

# plt.hist(back_yd, 60, color = 'black', label = 'background flow', weights = np.ones(Nsim)/Nsim, range = (min(back_yd),max(back_yd)))
# plt.hist(u[:,1],60, color = 'red', label = 'control action', alpha=.5, weights = np.ones(Nsim)/Nsim, range = (min(back_yd),max(back_yd)))

plt.hist(back_yd, 60, color = 'grey', weights = np.ones(Nsim)/Nsim, range = (min(back_yd),max(back_yd)))
plt.hist(u[:,1],60, color = 'red', alpha=.5, weights = np.ones(Nsim)/Nsim, range = (min(back_yd),max(back_yd)))
#plt.title('distribution of the y-component of velocities')
#plt.ylabel('probability of $u_y$')
#plt.ylim((0,1))
plt.xlim((-.3, .3))
plt.yticks([.1,.2])
plt.ylim((0,.3))
plt.xticks([-.1,-.3,.1,.3])

plt.tight_layout()

fig3 = plt.figure()
# plt.hist((back_xd**2+back_yd**2)**.5, 40, color = 'black', label = 'background flow', weights = np.ones(Nsim)/Nsim , range = (min((back_yd**2 + back_yd**2)**.5),max((back_yd**2 + back_yd**2)**.5)))
# plt.hist((u[:,0]**2+u[:,1]**2)**.5,40, color = 'red', label = 'control action', alpha = .5, weights = np.ones(Nsim)/Nsim, range = (min((back_yd**2 + back_yd**2)**.5),max((back_yd**2 + back_yd**2)**.5)))

plt.hist((back_xd**2+back_yd**2)**.5, 40, color = 'grey', weights = np.ones(Nsim)/Nsim , range = (min((back_yd**2 + back_yd**2)**.5),max((back_yd**2 + back_yd**2)**.5)))
plt.hist((u[:,0]**2+u[:,1]**2)**.5,40, color = 'red', alpha = .5, weights = np.ones(Nsim)/Nsim, range = (min((back_yd**2 + back_yd**2)**.5),max((back_yd**2 + back_yd**2)**.5)))
#plt.title('distribution of the $\||u\||$ of velocities')
#plt.ylabel('probability of $\||u\||$')
#plt.ylim((0,1))
plt.xticks(np.arange(.1,.7,.2))
plt.ylim((0,.42))
plt.xlim((0, np.pi/10*2**.5))
plt.tight_layout()

#%% heading direction plot
plt.rcParams.update({'font.size': 20})

back_hd = np.arctan(back_yd/back_xd)
#ctrl_hd = np.arctan(u[:,1]/u[:,0])
ctrl_hd = np.arctan(((x[1:,1] - x[:-1,1])/Delta)/((x[1:,0] - x[:-1,0])/Delta))


for i in range(Nsim):
    if back_xd[i] < 0 and back_yd[i] < 0:
        back_hd[i] += -np.pi
    if back_xd[i] < 0 and back_yd[i] > 0:
        back_hd[i] += np.pi
        

for i in range(Nsim):
    if u[i,0] < 0 and u[i,1] < 0:
        ctrl_hd[i] += -np.pi
    if u[i,0] < 0 and u[i,1] > 0:
        ctrl_hd[i] += np.pi
        
        
N = 100

ax = plt.subplot(111, projection='polar')
ax.hist(back_hd, 16, range = (-np.pi,np.pi), weights = np.ones(Nsim)/Nsim, width =.4, color = 'grey')
ax.hist(ctrl_hd, 16, range = (-np.pi,np.pi), weights = np.ones(Nsim)/Nsim, width =.4, color = 'red', alpha = .5)
ax.set_rticks([.1, .2, .3])
plt.tight_layout()

