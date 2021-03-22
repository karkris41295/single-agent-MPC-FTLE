#Double sweep of performance in R and T_horizon

import mpctools as mpc
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

d_sweep_data = []
for ths in range(1,21):
    t = []
    es = []
    cs = []
    preds = []
    tc = []
    es2 = []
    cs2 = []
    rq = []
    
    for runs in range(0, 100):
        
        # Define Parameters
        Q1 = 1
        Q = np.eye(Nx)*Q1
        Q2 = 0
        R1 = 10**(runs*.04 - 2)
        #R1 = 1
        R = R1*np.eye(Nu)
        ubot = .1
        #ubot = (runs/1000)+.0001
        Nt = int((1./Delta)*ths*.5)
        #Nt = runs
        
        goal = np.array([.5, .5, 0])
        Q[2,2] = 0
        
        def lfunc(x,u):
            """Standard quadratic stage cost."""
            return mpc.mtimes(u.T, R, u) + mpc.mtimes((x-goal).T, Q, (x-goal)) 
    
        l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")
        
        
        def Pffunc(x):
            """Penalty cost function"""
            return Q2*mpc.mtimes((x-goal).T,(x-goal))
    
        Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf") 
        
        # Bounds on u. Here, they are all [-1, 1]
        lb = {"u" : -ubot*np.ones((Nu,))}
        ub = {"u" : ubot*np.ones((Nu,))}
        
        # Make optimizers.
        xi = np.array([2,1,0])
        N = {"x":Nx, "u":Nu, "t":Nt}
    
        #This version with terminal penalty
        solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, Pf=Pf, x0=xi, lb=lb, ub=ub,verbosity=0)
        
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
        
            # Print stats.
            solver.saveguess()
            solver.fixvar("x",0,solver.var["x",1])
            
            #print("%d: %s" % (t,solver.stats["status"]))
            u += [np.array(solver.var["u",0,:]).flatten()]
            # calling solver variables. Can pull out predicted trajectories from here.
            pred += [solver.var["x",:,:]]
            
            # Simulate.
            x += [ocean.sim(x[t[runs]],u[t[runs]])]
            t[runs] += 1
            
        print("Param " + str(runs) + ", $T_H$ = " + str(ths))
        
        u = np.array(u)
        x = np.array(x)
        energy = (u[:,0]**2 + u[:,1]**2)**.5
        offshoot = ((x[:,0]-goal[0])**2 + (x[:,1]-goal[1])**2)**.5
        es += [energy]
        cs += [offshoot]
        
        energy2 = np.sum(energy)
        cost2 = np.sum(offshoot)
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
        rq += [R1]
        
    d_sweep_data += [[Nt,es,es2,goal,preds,tc,rq]]


#%% Saving data

import pickle 

with open ('swp1t11sp5natoq1guess', 'wb') as f:
    pickle.dump([A, eps, omega, Delta, Nsim, Nt, Nu, Nx, d_sweep_data], f)

