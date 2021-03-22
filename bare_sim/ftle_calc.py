# Calculated FTLE fields for later loading

import numpy as np
import matplotlib.pyplot as plt

# %% Paramteters from Shadden Physica D
A = .1
eps = .25
omega = 2*np.pi/10
Delta = .2
Nsim = 400 # if these values are changed, the values also need to be changed in sim_ftle.py

# %% Functions

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

dx = .005 #try .005
xvec = np.arange(0-dx, 2+dx, dx)
yvec = np.arange(0-dx, 1+dx, dx)

x0, y0 = np.meshgrid(xvec, yvec)
yIC = np.zeros((2, len(yvec), len(xvec)))
yIC[0], yIC[1] = x0, y0

# %% Calculate FTLE

dt = 0.025;  # timestep for integrator (try .005)
dt2 = Delta # timestep for frame
Tin = 15;     # duration of integration (maybe use 15) (forward time horiz)
T = Nsim*dt2; # total time over which simulation runs

solfor = []
solbac = []

for m in np.arange(0, T, dt2):
    
    # Forward time LCS
    yin_for = yIC
    
    for i in np.arange(0+m, Tin+m, dt):
        yout = rk4singlestep(lambda t, y: doublegyreVEC(t,y,A,eps,omega),dt,i,yin_for)
        yin_for = yout
    
    xT = yin_for[0]
    yT = yin_for[1]

    # Finite difference to compute the gradient
    dxTdx0, dxTdy0 = np.gradient(xT, dx, dx)
    dyTdx0, dyTdy0 = np.gradient(yT, dx, dx)

    D = np.eye(2)
    sigma = xT.copy()*0
    for i in range(len(xvec)):
        for j in range(len(yvec)):
            D[0,0] = dxTdx0[j,i];
            D[0,1] = dxTdy0[j,i];
            D[1,0] = dyTdx0[j,i];
            D[1,1] = dyTdy0[j,i];
            sigma[j,i] = abs((1./Tin)) * max(np.linalg.eigvals(np.dot(D.T, D)))
    
    sigma = (sigma - np.min(sigma))/(np.max(sigma) - np.min(sigma))
    solfor += [sigma]

    # Backward time LCS
    yin_bac = yIC
    
    for i in np.arange(0+m, -Tin+m, -dt):
        yout = rk4singlestep(lambda t, y: doublegyreVEC(t,y,A,eps,omega),-dt,i,yin_bac)
        yin_bac = yout
    
    xT = yin_bac[0]
    yT = yin_bac[1]

    # Finite difference to compute the gradient
    dxTdx0, dxTdy0 = np.gradient(xT, dx, dx)
    dyTdx0, dyTdy0 = np.gradient(yT, dx, dx)

    D = np.eye(2)
    sigma = xT.copy()*0
    for i in range(len(xvec)):
        for j in range(len(yvec)):
            D[0,0] = dxTdx0[j,i];
            D[0,1] = dxTdy0[j,i];
            D[1,0] = dyTdx0[j,i];
            D[1,1] = dyTdy0[j,i];
            sigma[j,i] = (1./Tin) * max(np.linalg.eigvals(np.dot(D.T, D)))
    
    sigma = (sigma - np.min(sigma))/(np.max(sigma) - np.min(sigma))
    solbac += [sigma]
    
    print("Time = " + str(m))
# %% Animation/Plotting
'''
recall = np.load('ftle_th12.npz')
data = [recall[key] for key in recall]
x0, y0, solfor, solbac = data
'''
from matplotlib import animation


fig = plt.figure()
ax = plt.subplot(111, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))
def update(num):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    ax.collections = []
    #solfor[num][solfor[num] > .1] = 1
    #solfor[num][solbac[num] > .1] = 1
    ax.contour(x0, y0, (solfor[num]), origin = 'lower', cmap = 'winter', alpha = 1)
    ax.contour(x0, y0, (solbac[num]), origin = 'lower', cmap = 'autumn', alpha = 1)

anim = animation.FuncAnimation(fig, update, blit=False)

# %% Storing Results

np.savez('ftle_th15.npz', x0, y0, solfor, solbac)
#%%
#recall = np.load('ftle_th15.npz')
#data = [recall[key] for key in recall]