# Loading Gulf of Mexico data
import matplotlib.pyplot as plt
import numpy as np
import netCDF4

#%% Load expt_19.0 (10/2/92-7/31/95)

source_92 = ('http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_19.0?lat[0:1:2000],lon[0:1:4499],time[0:1:1030]')
dwn92 = netCDF4.Dataset(source_92)

lat = dwn92['lat'][:] #'y' coordinates on map
lon = dwn92['lon'][:] #'x' coordinates on map
t_92 = dwn92['time'][:]

lon = lon[np.where((lon>=-98) * (lon<=-76.4))]
lat = lat[np.where((lat>=18.1) * (lat<=32))]

#%% Redefine source to include velocities

source_92 = ('http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_19.0?water_u[0:1:1030][1][1227:1:1400][1025:1:1295],water_v[0:1:1030][1][1227:1:1400][1025:1:1295]')
dwn92 = netCDF4.Dataset(source_92)

#%%
uvel = []
vvel = []
for i in range(len(t_92)):
    print('Loading ' + str(i) + '/' + str(len(t_92)))
    uvel += [dwn92['water_u'][i,0,:,:]]
    vvel += [dwn92['water_v'][i,0,:,:]]
    
#%% Animate data

uvec = np.ma.asarray(uvel)
vvec = np.ma.asarray(vvel)
from matplotlib import animation

fig, ax = plt.subplots(1,1)
qui = ax.quiver(lon, lat, uvec[0], vvec[0], color = 'grey')
ax.set_xlim(lon[0],lon[-1])
ax.set_ylim(lat[0],lat[-1])

def update(num,Q):
    
    ax.set_title('time = ' + str(num))
    Q.set_UVC(uvec[num], vvec[num])
    
    return Q,

anim = animation.FuncAnimation(fig, update, fargs=(qui,),interval=1, blit=False, repeat_delay = 10)

#%% Store data
#np.savez('gulf92.npz', lon,lat,t_92,uvec,vvec)

#%% https://pyproj4.github.io/pyproj/stable/api/proj.html#pyproj-proj

# geo_to_utm = pyproj.Proj('+proj=utm +zone=17R, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
# adcp_lon_utm, adcp_lat_utm = geo_to_utm(adcp_lon, adcp_lat)
