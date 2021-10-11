# ABC Double sweep

import mpctools as mpc
import numpy as np

#%% Parameters
A = 3**.5 * .1
B = 2**.5 * .1
C = 1 * .1
amp = 0.5 * .1
om = 2*np.pi/10

# %% Define model and parameters.
Delta = .1
Nx = 4
Nu = 3

def dgyre(x, u, A = A, B=B, C=C):
    """Continuous-time ODE model."""
    dxdt = [
        (A+amp*np.sin(om*x[3]))*np.sin(x[2]) + C*np.cos(x[1]) + u[0],
        B*np.sin(x[0]) + (A+amp*np.sin(om*x[3]))*np.cos(x[2]) + u[1],
        B*np.cos(x[0]) + C*np.sin(x[1]) + u[2],
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
for ths in range(1,10):
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
        R1 = runs
        #R1 = 1
        R = R1*np.eye(Nu)
        ubot = A+B+C
        #ubot = (runs/1000)+.0001
        Nt = int((1./Delta)*ths)
        #Nt = runs
        
        goal = np.array([5, 2, 1, 0])
        Q[-1,-1] = 0
        
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
        xi = np.array([np.pi/2,1,6,0])
        N = {"x":Nx, "u":Nu, "t":Nt}
    
        #This version with terminal penalty
        solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, Pf=Pf, x0=xi, lb=lb, ub=ub,verbosity=0)
        
        Nsim = 2000
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
            #x += [ocean.sim(x[t[runs]],u[t[runs]])]
            t[runs] += 1
            
        
        print("Param " + str(runs) + ", $T_H$ = " + str(ths))
        
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
        
        x = pred3[:,0,:]
        
        preds += [pred3]
        rq += [R1]
        
        u = np.array(u)
        x = np.array(x)
        energy = (u[:,0]**2 + u[:,1]**2 + u[:,2]**2)**.5
        offshoot = ((x[:,0]-goal[0])**2 + (x[:,1]-goal[1])**2 + (x[:,2]-goal[2])**2)**.5
        es += [energy]
        cs += [offshoot]
        
        energy2 = np.sum(energy)
        cost2 = np.sum(offshoot)
        cs2 += [cost2]
        es2 += [energy2]
        tc += [cost2 + R1*energy2]
        
        
    d_sweep_data += [[Nt,es,es2,goal,preds,tc,rq,x,u]]


#%% Saving data


# import pickle 

# with open ('swp1t10abc_ldg', 'wb') as f:
#     pickle.dump([A, B, C, Delta, Nsim, Nt, Nu, Nx, d_sweep_data], f)



