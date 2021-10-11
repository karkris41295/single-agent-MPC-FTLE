# Calculating LCS for ABC flow

import numpy as np
import matplotlib.pyplot as plt

# %% Paramteters 
A = 3**.5 * .1
B = 2**.5 * .1
C = 1 * .1
amp = 0.05
om = 2*np.pi/10

Delta = .2
Nsim = 50 # if these values are changed, the values also need to be changed in sim_ftle.py

# %% Functions

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
    
def rk4singlestep(fun,dt,t0,y0):
    f1 = fun(t0,y0);
    f2 = fun(t0+dt/2,y0+(dt/2)*f1);
    f3 = fun(t0+dt/2,y0+(dt/2)*f2);
    f4 = fun(t0+dt,y0+dt*f3);

    yout = y0 + (dt/6)*(f1+2*f2+2*f3+f4)
    return yout

#%% Part 1 - Initialize grid of particles through vector field

dx = .05 #try .005
xvec = np.arange(0, 2*np.pi+dx, dx)
yvec = np.arange(0, 2*np.pi+dx, dx)
zvec = np.arange(0, 2*np.pi+dx, dx)

x0, y0, z0 = np.meshgrid(xvec, yvec, zvec)
yIC = np.zeros((3 ,len(zvec), len(yvec), len(xvec)))
yIC[0], yIC[1], yIC[2] = x0, y0, z0

# %% Calculate FTLE

dt = 0.05;  # timestep for integrator (try .005)
dt2 = Delta # timestep for frame
Tin = 40;     # duration of integration (maybe use 15) (forward time horiz)
T = Nsim*dt2; # total time over which simulation runs

solfor = []
solbac = []

for m in np.arange(0, T, dt2):
    
    # Forward time LCS
    yin_for = yIC
    
    for i in np.arange(0+m, Tin+m, dt):
        yout = rk4singlestep(lambda t, y: doublegyreVEC(t,y,A,B,C),dt,i,yin_for)
        yin_for = yout
    
    xT = yin_for[0]
    yT = yin_for[1]
    zT = yin_for[2]

    # Finite difference to compute the gradient
    dxTdx0, dxTdy0, dxTdz0 = np.gradient(xT, dx, dx, dx)
    dyTdx0, dyTdy0, dyTdz0 = np.gradient(yT, dx, dx, dx)
    dzTdx0, dzTdy0, dzTdz0 = np.gradient(yT, dx, dx, dx)

    D = np.eye(3)
    sigma = xT.copy()*0
    for i in range(len(xvec)):
        for j in range(len(yvec)):
            for k in range(len(zvec)):
                D[0,0] = dxTdx0[k,j,i];
                D[0,1] = dxTdy0[k,j,i];
                D[0,2] = dxTdz0[k,j,i]; #
                
                D[1,0] = dyTdx0[k,j,i];
                D[1,1] = dyTdy0[k,j,i];
                D[1,2] = dyTdz0[k,j,i]; #
                
                D[2,0] = dzTdx0[k,j,i];
                D[2,1] = dzTdy0[k,j,i];
                D[2,2] = dzTdz0[k,j,i]; #
                sigma[k,j,i] = abs((1./Tin)) * max(np.linalg.eigvals(np.dot(D.T, D)))
        
    sigma = (sigma - np.min(sigma))/(np.max(sigma) - np.min(sigma))
    solfor += [sigma]

    # Backward time LCS
    yin_bac = yIC
    
    for i in np.arange(0+m, -Tin+m, -dt):
        yout = rk4singlestep(lambda t, y: doublegyreVEC(t,y,A,B,C),-dt,i,yin_bac)
        yin_bac = yout
    
    xT = yin_bac[0]
    yT = yin_bac[1]

    # Finite difference to compute the gradient
    dxTdx0, dxTdy0, dxTdz0 = np.gradient(xT, dx, dx, dx)
    dyTdx0, dyTdy0, dyTdz0 = np.gradient(yT, dx, dx, dx)
    dzTdx0, dzTdy0, dzTdz0 = np.gradient(yT, dx, dx, dx)
    
    D = np.eye(3)
    sigma = xT.copy()*0
    for i in range(len(xvec)):
        for j in range(len(yvec)):
            for k in range(len(zvec)):
                D[0,0] = dxTdx0[k,j,i];
                D[0,1] = dxTdy0[k,j,i];
                D[0,2] = dxTdz0[k,j,i]; #
                
                D[1,0] = dyTdx0[k,j,i];
                D[1,1] = dyTdy0[k,j,i];
                D[1,2] = dyTdz0[k,j,i]; #
                
                D[2,0] = dzTdx0[k,j,i];
                D[2,1] = dzTdy0[k,j,i];
                D[2,2] = dzTdz0[k,j,i]; #
                sigma[k,j,i] = abs((1./Tin)) * max(np.linalg.eigvals(np.dot(D.T, D)))
    
    sigma = (sigma - np.min(sigma))/(np.max(sigma) - np.min(sigma))
    solbac += [sigma]
    
    print("Time = " + str(m))
# %% Animation/Plotting
# '''
# recall = np.load('ftle_th12.npz')
# data = [recall[key] for key in recall]
# x0, y0, solfor, solbac = data
# '''
# from matplotlib import animation

from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib import animation

fig = plt.figure()
ax = fig.gca(projection='3d')

x02, y02 = np.meshgrid(x0[0,:,0], x0[0,:,0])

def update(num,Q):
    #global Q2
    #t_snap = Delta*num
    ax.collections = []
    ax.contour(x02, y02, solfor[num%50][:,:,0], zdir='z', offset=0, cmap='winter')
    ax.contour(x02, y02, solbac[num%50][:,:,0], zdir='z', offset=0, cmap='autumn')

    
#anim = animation.FuncAnimation(fig, update, fargs=(qui,),interval=100, blit=False, repeat_delay = 10)
    
anim = animation.FuncAnimation(fig, update, fargs=(1,),interval=100, blit=False, repeat_delay = 10)

# %% Storing Results

#np.savez('ftle_th15.npz', x0, y0, solfor, solbac)
#np.savez('ftle_abc_th8hi_dg.npz', x0, y0,z0, solfor, solbac)
#%%
#recall = np.load('ftle_th15.npz')
#data = [recall[key] for key in recall]