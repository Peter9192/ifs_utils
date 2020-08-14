""" Functions to process ECMWF ERA5 model level data

- get_reference_levels(levs):
    Return the reference pressure for level index coordinates.

- pressure_funcs(ph, levs):
    Variables related to the vertical discretization in IFS.
    returns: dp, dlnp, upper, alpha

- pressure_at_model_levels(levs, psfc):
    Compute the pressure at half and full model levels.

- geopotential_at_model_levels(dlnp, Tv, Z0, alpha, levs):
    Calculate geop. for each level (ecwmf documentation Pt 3, eq 2.21).

- geostrophic_wind_at_model_levels(lat, Z, Tv, dp, dlnp, alpha, upper):
    Compute geostrophic wind at model levels.
- virtual_temperature(T,q):
    IFS documenation, part III page 5

- potential_temperature(T,p):
    Compute potential temperature

- surface_geopotential_hypso(psfc,mslp,Tv_sfc):
    Workaround to calculate surface geopotential (didn't download it).

TODO: I made some quick and dirty adaptations to pass variable-array input
      and flexible type input, but this should be done more consistently.
"""

import xarray as xr
import pandas as pd
import numpy as np
import os

# Constants
Rd = 287. # Gas constant for dry air (J/K/kg)
Rv = 461. # Gas constant for water vapour (J/K/kg)
cp = 1006. # Heat capacity of air at constant pressure (J/K/kg)
g = 9.81 # Accelleration due to gravity (m/s)
a = 6371000. # Radius of earth (m)
Omega = 7.292e-5

def get_reference_levels(levs):
    """ Return the reference pressure for level index coordinates."""
    thisdir = os.path.dirname(__file__)
    l137 = pd.read_csv(thisdir+'/ecmwf_l137.txt', index_col=0)
    return l137.z[levs].values

def pressure_funcs(ph, levs, z_axis=1):
    """ Variables related to the vertical discretization in IFS. """
    # Move z_axis to front
    _ph = np.moveaxis(ph, z_axis, 0)

    if levs[0]<levs[1]: # IFS upside down format
        dp = np.diff(_ph, axis=0)
        dlnp = np.log(_ph[1:, ...]/_ph[:-1, ...])
        upper = _ph[:-1, ...]
    else: # Sometimes I manually inverse the levels beforehand.
        dp = -np.diff(_ph, axis=0)
        dlnp = np.log(_ph[:-1, ...]/_ph[1:, ...])
        upper = _ph[1:, ...]

    alpha = 1-upper/dp*dlnp
    return [np.moveaxis(a, 0, z_axis) for a in [dp, dlnp, upper, alpha]]

def pressure_at_model_levels(levs, psfc, z_axis=1):
    """ Compute the pressure at half and full model levels. """

    # Pressure is defined at cell edges, so need one extra to interpolate
    if levs[1]>levs[0]:
        levs = np.insert(levs,0,levs[0]-1)
    else:
        levs = np.append(levs,levs[-1]-1)

    # Get a and b values from level definitions
    # thisdir = os.path.dirname(__file__) # old, didn't work with links
    thisdir = os.path.dirname(os.path.realpath(__file__))

    l137 = pd.read_csv(thisdir+'/ecmwf_l137.txt', index_col=0)
    a = l137.a[levs]
    b = l137.b[levs]

    # Compute pressure at half and full levels
    s1 = [slice(None)]+[None]*psfc.ndim # [:, None, None, None ...]
    s2 = [None]+[slice(None)]*psfc.ndim # [None, :, :, :, :, :, : ...]
    ph = a[tuple(s1)]+b[tuple(s1)]*psfc.values[tuple(s2)]
    pf = (ph[:-1, ...] + ph[1:, ...])*.5

    # Old:
    # ph = a[:, ...]+b[None,:,None,None]*psfc.values[None, ...]
    # pf = (ph[:,:-1,:,:] + ph[:,1:,:,:])*.5
    return [np.moveaxis(a, 0, z_axis) for a in [ph, pf]]

def geopotential_at_model_levels(dlnp, Tv, Z0, alpha, levs, z_axis=1, t_axis=0):
    """ Calculate geop. for each level (ecwmf documentation Pt 3, eq 2.21). """

    # Move z_axis to front
    _dlnp = np.moveaxis(dlnp, z_axis, 0)
    _Tv = np.moveaxis(Tv, z_axis, 0)
    _alpha = np.moveaxis(alpha, z_axis, 0)
    t_axis = t_axis+1 if t_axis < z_axis else t_axis

    # Geopotential jump over each layer (hypsometric formula)
    dZ = Rd*_Tv*_dlnp

    # However, IFS stores the coordinates upside down...
    # In that case, we must flip the array
    if levs[0]<levs[1]:
        dZ = dZ[::-1, ...]

    # Accumulate layer geopotentials starting from the surface
    sl = [slice(None)]*dZ.ndim     # [None,None,:,:] for t,z,y,x
    sl[0] = None # expand dimensions along z_axis
    sl[t_axis] = None # expand dimensions along t_axis
    Zhalf = Z0[tuple(sl)] + np.cumsum(dZ, axis=0)
    Zhalf = np.insert(Zhalf, 0, Z0[tuple(sl)], axis=0)

    # Back-flip
    if levs[0]<levs[1]:
        Zhalf = Zhalf[::-1, ...]

    # Compute full level geopotentials
    if levs[0]<levs[1]: # IFS upside down format
        Z = Zhalf[1:, ...] + _alpha*Rd*_Tv
    else:
        Z = Zhalf[:-1, ...] + _alpha*Rd*_Tv

    return np.moveaxis(Z, 0, z_axis)

def geostrophic_wind_at_model_levels(lat, Z, Tv, dp, dlnp, alpha, upper):
    """ Compute geostrophic wind at model levels. """
    dx = 0.3*np.pi/180.  # grid spacing in radians lat/lon
    dy = -0.3*np.pi/180. # latitude decrease along axis
    f = 2*Omega*np.sin(lat*np.pi/180.)[None, None, :, None] # Coriolis parameter (1/s)
    mapfac = np.cos(lat*np.pi/180.)[None, None, :, None] # map scale factor

    # Isobaric terms
    Vg_iso = np.gradient(Z, dx, axis=3)
    Ug_iso = np.gradient(Z, dy, axis=2)

    # Non-isobaric terms (see IFS docs part III eq. 2.24)
    Vg_non_iso = Rd*Tv/dp*(dlnp*np.gradient(upper, dx, axis=3)
                           +alpha*np.gradient(dp, dx, axis=3))
    Ug_non_iso = Rd*Tv/dp*(dlnp*np.gradient(upper, dy, axis=2)
                           +alpha*np.gradient(dp, dy, axis=2))

    # Combined etc.
    Vg =  1/(a*f)*(Vg_iso + Vg_non_iso)
    Ug = -1/(a*f)*(Ug_iso + Ug_non_iso)*mapfac
    return xr.Dataset({'ug':Ug, 'vg':Vg})

def momentum_tendencies_at_ml(lat, Z, Tv, dp, dlnp, alpha, upper, u, v):
    """ Compute tendencies in IFS momentum equations (all terms to rhs)."""

    dx = 0.3*np.pi/180.  # grid spacing in radians lat/lon
    dy = -0.3*np.pi/180. # latitude decrease along axis
    f = 2*Omega*np.sin(lat*np.pi/180.)[None, None, :, None] # Coriolis parameter (1/s)
    mapfac = np.cos(lat*np.pi/180.)[None, None, :, None] # map scale factor

    adv_u = -1/(a*mapfac**2)*(u*np.gradient(u, dx, axis=3) + mapfac*v*np.gradient(u, dy, axis=2))
    adv_v = -1/(a*mapfac**2)*(u*np.gradient(v, dx, axis=3) + mapfac*v*np.gradient(v, dy, axis=2))
    # cur_v = -1/(a*mapfac**2*np.sin(theta*np.pi/180.)[None, None, :, None]*(u**2+v**2)
    # sub_u = See eq 3.6 in the IFS documentation
    # sub_v =
    cor_u = f*v
    cor_v = -f*u
    pgf_u =  -1/a*(np.gradient(Z, dx, axis=3) + Rd*Tv/dp*
        (dlnp*np.gradient(upper, dx, axis=3)+alpha*np.gradient(dp, dx, axis=3)))
    pgf_v =  -mapfac/a*(np.gradient(Z, dy, axis=2) + Rd*Tv/dp*
        (dlnp*np.gradient(upper, dy, axis=2)+alpha*np.gradient(dp, dy, axis=2)))
    # par_u = Currently not considered
    # par_v =

    return xr.Dataset({'adv_u':adv_u, 'adv_v':adv_v, 'cor_u':cor_u,
                       'cor_v':cor_v, 'pgf_u':pgf_u, 'pgf_v':pgf_v})

def virtual_temperature(T,q):
    """ IFS documenation, part III page 5 """
    return T*(1+(Rv/Rd-1)*q)

def potential_temperature(T,p):
    pref = 100000 # Pa
    return T*(pref/p)**(Rd/cp)

def surface_geopotential_hypso(psfc,mslp,Tv_sfc):
    """ Workaround to calculate surface geopotential (didn't download it). """
    eps = 0.61

    # Hypsometric equation
    Z0 = np.log(mslp/psfc)*Rd*Tv_sfc.values

    # Time averaging for a better estimate
    return Z0.mean(dim='time')

if __name__=="__main__":
    sfc = xr.open_dataset('era5_201006_sfc.nc')
    era5 = xr.open_dataset('era5_201006_ml.nc')#.sel(level=slice(None,124,-1))

    lons = era5.longitude.values
    lats = era5.latitude.values
    levs = era5.level.values

    ph, pf = pressure_at_model_levels(levs, sfc.sp)
    dp, dlnp, upper, alpha = pressure_funcs(ph, levs)
    Tv = virtual_temperature(era5.t, era5.q)
    Z0 = surface_geopotential_hypso(sfc.sp,sfc.msl,Tv.sel(level=137))
    Z = geopotential_at_model_levels(dlnp, Tv, Z0, alpha, levs)
    z = Z/9.81
    Ug, Vg = geostrophic_wind_at_model_levels(lats, Z, Tv, dp, dlnp, alpha, upper)
    Ua = Ug - era5.u
    Va = Vg - era5.v
    Ub = -np.gradient(Ug, axis=1)/np.gradient(pf,axis=1)
    Vb = -np.gradient(Vg, axis=1)/np.gradient(pf,axis=1)

    # Full wind
    magf = (era5.u**2+era5.v**2)**.5
    dirf = np.arctan2(-era5.u,-era5.v)*180/np.pi%360
    # Geostrophic
    magg = (Ug**2+Vg**2)**.5
    dirg = np.arctan2(-Ug,-Vg)*180/np.pi%360
    # Ageostrophic
    maga = (Ua**2+Va**2)**.5
    dira = np.arctan2(-Ua,-Va)*180/np.pi%360
    # Thermal wind gradient
    magb = (Ub**2+Vb**2)**.5
    dirb = np.arctan2(-Ub,-Vb)*180/np.pi%360

    import matplotlib.pyplot as plt
    # Some phase space plots
    plt.hexbin(magf.values.ravel(),dirf.values.ravel())
    plt.xlabel('full wind magnitude (m/s)')
    plt.ylabel('full wind direction (deg)')
    plt.savefig('fig0')
    plt.close()

    plt.hexbin(magb.ravel(),dirb.ravel(), extent=(0,0.001,0,360))
    plt.xlabel('thermal wind gradient magnitude (m/s/Pa)')
    plt.ylabel('thermal wind gradient direction (deg/Pa)')
    plt.savefig('fig1')
    plt.close()

    plt.hexbin(magf.values.ravel(),magb.ravel(), extent=(0,25,0,0.001))
    plt.xlabel('full wind speed (m/s)')
    plt.ylabel('thermal wind gradient magnitude (m/s/Pa)')
    plt.savefig('fig2')
    plt.close()

    plt.hexbin(dirf.values.ravel(),dirb.ravel())
    plt.xlabel('full wind direction (degrees)')
    plt.ylabel('thermal wind gradient direction (degrees/Pa)')
    plt.savefig('fig3')
    plt.close()

    plt.hexbin(dirg.values.ravel(),dira.values.ravel())
    plt.xlabel('full wind direction (degrees)')
    plt.ylabel('thermal wind gradient direction (degrees/Pa)')
    plt.savefig('fig3')
    plt.close()

    # # Testing
    # import matplotlib.pyplot as plt
    # from ncview import Player
    # fig, ax = plt.subplots()
    # vmin = sfc.msl.min()
    # vmax = sfc.msl.max()
    # qm = plt.pcolormesh(lons, lats, sfc.msl.isel(time=0), vmin=vmin, vmax=vmax)
    # q = plt.quiver(lons, lats, Ug.isel(time=0,level=0), Vg.isel(time=0,level=0))
    #
    # def init():
    #     qm.set_array([])
    #     q.set_UVC([],[])
    #     return qm, q
    #
    # def update(i):
    #     qm.set_array(sfc.msl.values[i,:-1,:-1].ravel())
    #     q.set_UVC(Ug.values[i,0,:,:], Vg.values[i,0,:,:])
    #     return qm,q
    #
    # nframes = len(Ug)-1
    # anim = Player(fig, update, nframes, init_func=init, blit=True)
    # plt.show()

    #
    # # Compute wind speed, baroclinity, etc.
    # theta = era5.t*(pf/1000)**0.285
    # thetav = theta*(1+0.61*era5.q)
    # rho = pf*100/(287.058*tv)
    # wspd = (era5.u**2+era5.v**2)**.5
    #
    #
    # # Interpolate variables to fixed true horizontal surfaces
    # znew = np.arange(0,610,5)[None,:,None,None]
    # nznew = znew.shape[1]
    # thetav_fixed_z = interp_along_axis(thetav.values, z, znew, axis=1, method='linear')
    # values = np.ma.masked_invalid(thetav_fixed_z)
    #
    # # Calculate virtual potential temperature gradient (example for plotting)
    # tvfz_dz = np.ma.masked_invalid(np.gradient(values, 5, axis=1))
    # dx = 0.3*np.pi/180.*6371000*np.sin(lats[20]/180*np.pi) #(1 degree --> 1 m)
    # tvfz_dx = np.ma.masked_invalid(np.gradient(values,dx, axis=3))
