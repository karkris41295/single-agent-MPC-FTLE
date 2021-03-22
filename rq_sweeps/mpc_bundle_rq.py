#Evolving MPC Horizon bundles with R/Q for the DGyre

import mpctools as mpc
import matplotlib.pyplot as plt
import numpy as np

# %% Paramteters from Shadden Physica D
A = .1
eps = .25
omega = 2*np.pi/10

# %% Define model and parameters.
Delta = .1
Nx = 3
Nu = 2

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
# %% Now simulate.

t = []
es = []
preds = []
tc = []
es2 = []
cs2 = []

for runs in range(0, 100): # range specifies the number of different parameter values to sweep through
    
    # Define Parameters
    Q = np.eye(Nx)
    Q[2,2] = 0
    R1 = (runs) # sweeping through R here
    #R1 = 1
    R = R1*np.eye(Nu)
    ubot = .1
    #ubot = (runs/1000)+.0001
    Nt = 20 #Delta*Nt gives the length of the time horizon, here it is 2
    #Nt = runs
    
    goal = np.array([.5, .5, 0])

    def lfunc(x,u):
        """Standard quadratic stage cost."""
        return mpc.mtimes(u.T, R, u) + mpc.mtimes((x-goal).T, Q, (x-goal)) 

    l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")
    
    
    def Pffunc(x):
        """Penalty cost function"""
        return 0*mpc.mtimes((x-goal).T,(x-goal))

    Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf") 
    
    # Bounds on u. Here, they are all [-1, 1]
    lb = {"u" : -ubot*np.ones((Nu,))}
    ub = {"u" : ubot*np.ones((Nu,))}
    
    # Make optimizers.
    xi = np.array([2,1,0])
    N = {"x":Nx, "u":Nu, "t":Nt}
    
    #version without a penalty
    solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, x0=xi, lb=lb, ub=ub,verbosity=0)

    #This version with terminal penalty
    #solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, Pf=Pf, x0=xi, lb=lb, ub=ub,verbosity=0)
    
    Nsim = 800
    x = []
    x += [xi]
    u =[]
    pred = []
    
    t += [0]
    hit = []
    times = []
    # Fix initial state.
    solver.fixvar("x", 0, x[t[runs]])  
    while (t[runs] < Nsim):
        
        # Solve nlp.
        solver.solve()   
    
        # Save guess and start from prev guess
        solver.saveguess()
        solver.fixvar("x",0,solver.var["x",1])
        # Print stats.
        #print("%d: %s" % (t,solver.stats["status"]))
        u += [np.array(solver.var["u",0,:]).flatten()]
        # calling solver variables. Can pull out predicted trajectories from here.
        pred += [solver.var["x",:,:]]
        
        # Simulate.
        x += [ocean.sim(x[t[runs]],u[t[runs]])]
        t[runs] += 1
        
    print("Param " + str(runs))
    
    u = np.array(u)
    x = np.array(x)
    energy = (u[:,0]**2 + u[:,1]**2)**.5
    es += [energy]
    
    energy2 = np.sum(energy)
    cost2 = np.sum(((x[:,0]-goal[0])**2 + (x[:,1]-goal[1])**2)**.5)
    cs2 += [cost2]
    es2 += [energy2]
    tc += [cost2 + R1*energy2]
    
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
    
    preds += [pred3]
    
# %% Moving bundle plot

from matplotlib import animation

qr = np.arange(0,100,1)
fig = plt.figure()
ax = plt.subplot(211, aspect = 'equal', xlim=(0, 2), ylim=(0, 1), title = 'MPC horizon bundle, $T_{horizon} = $' + str(Nt*Delta) + ', $u_{botmax} = $' + str(ubot))
ax2 = plt.subplot(212, title = 'costs', xlabel = 'R/Q', xlim=(0, max(qr)))
plt.tight_layout()
#qr_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

uru = np.array(es2)*qr
ax2.plot(qr,uru, 'grey', alpha = .3)
ax2.plot(qr,cs2, 'grey', alpha = .3)
ax2.plot(qr,tc, 'grey', alpha = .3)
particle1, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
particle2, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
particle3, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
traj1, = ax2.plot([],[], color='blue', alpha=0.6, label = '$\sum (u^TRu)$')
traj2, = ax2.plot([],[], color='red', alpha=0.6, label = '$\sum (x^TQx)$')
traj3, = ax2.plot([],[], color='purple', alpha=0.6, label = '$\sum (x^TQx + u^TRu)$')
ax2.legend()

from matplotlib import cm

def update(num):
    '''
    updates animation
    '''
    ax.cla()
    ax.set_xlim(0,2)
    ax.set_ylim(0,1)
    ax.set_title('MPC horizon bundle, $T_{horizon} = $' + str(Nt*Delta) + ', $u_{botmax} = $' + str(ubot))
    qr_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    ax.scatter(goal[0], goal[1], color = 'purple')
    e = es[num]
    for i in range(Nsim):
        ax.plot(preds[num][i,:2,0], preds[num][i,:2,1], alpha=0.5, c = cm.plasma(e[i]/max(1.00001*e)))
        
    qr_text.set_text('$R/Q = $' + str(qr[num]))
    
    particle1.set_data(qr[num], uru[num])
    traj1.set_data(qr[:num+1],uru[:num+1]) 
    particle2.set_data(qr[num], cs2[num])
    traj2.set_data(qr[:num+1],cs2[:num+1]) 
    particle3.set_data(qr[num], tc[num])
    traj3.set_data(qr[:num+1],tc[:num+1]) 
    
anim = animation.FuncAnimation(fig, update, blit=False, interval=100)

#%% Saving data
#np.savez('t2up1delp1guess.npz', A, eps, omega, Delta, Nsim, Nt, Nu, Nx, es, es2, goal, preds, qr, tc, ubot, uru)