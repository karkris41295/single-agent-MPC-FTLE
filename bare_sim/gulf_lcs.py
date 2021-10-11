# LCS on gulf of Mexico
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

#%%

recall = np.load('/Users/kartikkrishna/Downloads/bigdata/hycom/1992/gulf92.npz')
data = [recall[key] for key in recall]
lon,lat,t_92,uvec,vvec = data

t_days = np.arange(0,len(t_92),1)
uvec = uvec * 86.4; vvec = vvec*86.4

uvecint = uvec.copy()
vvecint = vvec.copy()
uvec = ma.masked_greater(abs(uvec),1e4)
vvec = ma.masked_greater(abs(vvec),1e4)

x02 = lon * 111 # converting to km
y02 = lat * 111

#%%
from matplotlib import animation

fig, ax = plt.subplots(1,1)
qui = ax.quiver(x02, y02, uvec[0], vvec[0], color = 'grey')
ax.set_xlim(x02[0],x02[-1])
ax.set_ylim(y02[0],y02[-1])

def update(num,Q):
    
    ax.set_title('time = ' + str(num))
    Q.set_UVC(uvec[num], vvec[num])
    
    return Q,

anim = animation.FuncAnimation(fig, update, fargs=(qui,),interval=1, blit=False, repeat_delay = 10)

#%%
from scipy.interpolate import RegularGridInterpolator

uvecint[abs(uvecint)>1e4] = 0; vvecint[abs(vvecint)>1e4] = 0
ufield = RegularGridInterpolator((t_days, y02, x02), uvecint, bounds_error = False, fill_value = 0)
vfield = RegularGridInterpolator((t_days, y02, x02), vvecint, bounds_error = False, fill_value = 0)

#%%

Delta = 1
Nsim = 50

# %% Functions

def doublegyreVEC(t, yin):
    
    sh = np.shape(yin[0]) 
    x = yin[0].flatten()
    y = yin[1].flatten()
    t = np.zeros(len(x)) + t
    
    pts = np.array([t,y,x]).T

    u = np.zeros(x.shape); 
    v = u.copy()

    u = ufield(pts);
    v = vfield(pts);
    
    u = u.reshape(sh[0],sh[1])
    v = v.reshape(sh[0],sh[1])

    return np.array([u,v])
    
def rk4singlestep(fun,dt,t0,y0):
    f1 = fun(t0,y0);
    f2 = fun(t0+dt/2,y0+(dt/2)*f1);
    f3 = fun(t0+dt/2,y0+(dt/2)*f2);
    f4 = fun(t0+dt,y0+dt*f3);

    yout = y0 + (dt/6)*(f1+2*f2+2*f3+f4)
    return yout

#%% Part 1 - Initialize grid of particles through vector field

dx = .7 #try .005
#xvec = np.arange(min(x02), max(x02), dx)
xvec = np.arange(-9510, -9210, dx)

#yvec = np.arange(min(y02), max(y02), dx)
yvec = np.arange(2080, 2230, dx)

x0, y0 = np.meshgrid(xvec, yvec)
yIC = np.zeros((2, len(yvec), len(xvec)))
yIC[0], yIC[1] = x0, y0

# %% Calculate FTLE

dt = 0.1;  # timestep for integrator (try .005)
dt2 = Delta # timestep for frame
Tin = 40;     # duration of integration (maybe use 15) (forward time horiz)
T = Nsim*dt2; # total time over which simulation runs

solfor = []
solbac = []

for m in np.arange(0, T, dt2):
    
    # Forward time LCS
    yin_for = yIC
    
    for i in np.arange(0+m, Tin+m, dt):
        yout = rk4singlestep(lambda t, y: doublegyreVEC(t,y),dt,i,yin_for)
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
        yout = rk4singlestep(lambda t, y: doublegyreVEC(t,y),-dt,i,yin_bac)
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
    
#%%

fig = plt.figure()

ax = plt.subplot(111)

def update(num,Q):
    #global Q2
    #t_snap = Delta*num
    ax.collections = []
    ax.contour(x0, y0, solfor[num], origin = 'lower', cmap = 'winter', alpha = 1)
    ax.contour(x0, y0, solbac[num], origin = 'lower', cmap = 'autumn', alpha = 1)


#anim = animation.FuncAnimation(fig, update, fargs=(qui,),interval=100, blit=False, repeat_delay = 10)
    
anim = animation.FuncAnimation(fig, update, fargs=(1,),interval=10, blit=False, repeat_delay = 10)

#%% Save

np.savez('ftle_th40_gulf_zoom1.npz', x0, y0, solfor, solbac)

