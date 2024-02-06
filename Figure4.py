
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator



tq                     = xr.open_dataset("../data/profile.nc")
txm                    = np.mean(tq.t_x, axis=2)
tym                    = np.mean(tq.t_y, axis=2)
qx                     = np.array(tq.q_x)
qy                     = np.array(tq.q_y)
ux                     = np.array(tq.u_x)
uy                     = np.array(tq.u_y)
vx                     = np.array(tq.v_x)
vy                     = np.array(tq.v_y)
vorx                   = np.array(tq.vor_x)
vory                   = np.array(tq.vor_y)
tx                     = tq.t_x
ty                     = tq.t_y
ztq                    = tq.ztq
zuv                    = tq.zuv


fn            = '../avgvor-mvs.csv'
vor           = np.array(pd.read_csv(fn))
mv = []
nonmv = []
for i in np.arange(884):
    if vor[i,0] > 0.0001:
        mv.append(i)
    else:
        nonmv.append(i)

del(tq)

for i in np.arange(161):
    tx[:,:,i]          = tx[:,:,i]-txm
    ty[:,:,i]          = ty[:,:,i]-tym





tx_all        = np.mean(tx, axis=0)
tx_ex         = np.mean(tx[nonmv,:,:], axis=0)
tx_mv         = np.mean(tx[mv,:,:], axis=0)
ty_all        = np.mean(ty, axis=0)
ty_ex         = np.mean(ty[nonmv,:,:], axis=0)
ty_mv         = np.mean(ty[mv,:,:], axis=0)
data1 = [tx_all, tx_ex, tx_mv, ty_all, ty_ex, ty_mv]

qx_all        = np.mean(qx, axis=0)
qx_ex         = np.mean(qx[nonmv,:,:], axis=0)
qx_mv         = np.mean(qx[mv,:,:], axis=0)
qy_all        = np.mean(qy, axis=0)
qy_ex         = np.mean(qy[nonmv,:,:], axis=0)
qy_mv         = np.mean(qy[mv,:,:], axis=0)
data2 = [qx_all, qx_ex, qx_mv, qy_all, qy_ex, qy_mv]

ux_all        = np.mean(ux, axis=0)
ux_ex         = np.mean(ux[nonmv,:,:], axis=0)
ux_mv         = np.mean(ux[mv,:,:], axis=0)
uy_all        = np.mean(uy, axis=0)
uy_ex         = np.mean(uy[nonmv,:,:], axis=0)
uy_mv         = np.mean(uy[mv,:,:], axis=0)

vx_all        = np.mean(vx, axis=0)
vx_ex         = np.mean(vx[nonmv,:,:], axis=0)
vx_mv         = np.mean(vx[mv,:,:], axis=0)
vy_all        = np.mean(vy, axis=0)
vy_ex         = np.mean(vy[nonmv,:,:], axis=0)
vy_mv         = np.mean(vy[mv,:,:], axis=0)
data3 = [ux_all, ux_ex, ux_mv, vy_all, vy_ex, vy_mv]

vorx_all        = np.mean(vorx, axis=0)
vorx_ex         = np.mean(vorx[nonmv,:,:], axis=0)
vorx_mv         = np.mean(vorx[mv,:,:], axis=0)
vory_all        = np.mean(vory, axis=0)
vory_ex         = np.mean(vory[nonmv,:,:], axis=0)
vory_mv         = np.mean(vory[mv,:,:], axis=0)
data4 = [vorx_all, vorx_ex, vorx_mv, vory_all, vory_ex, vory_mv]


plt.rc('font',family='Times New Roman')
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fig, ax       = plt.subplots(2, 3, figsize=(9, 6.5))
axs           = ax.flatten()

x             = np.arange(-40,40.5,0.5)
y             = np.arange(-40,40.5,0.5)


for i in np.arange(6):
    p1            = axs[i].contourf(x,zuv*0.001,10**4*data4[i], levels=np.arange(-5,5.5,0.5),cmap="RdBu_r")
    p2            = axs[i].contour(x,ztq*0.001,data1[i], levels=np.arange(-0.4, 0, 0.2),zorder=1, colors='darkblue')
    p3            = axs[i].contour(x,ztq*0.001,data2[i], levels=np.arange(10,18,2),zorder=2, colors='darkgreen')
    axs[i].clabel(p2, fontsize=8, inline=True)
    axs[i].clabel(p3, fontsize=8, inline=True)

    axs[i].set_xlabel('X (km)')
    if i >= 3:
        axs[i].set_xlabel('Y (km)')

    axs[i].set_ylabel('Height (km)')

cbar = fig.colorbar(p1, ax=axs,ticks=[-5,-2.5,0,2.5,5], shrink=0.4, orientation='horizontal', extend='both')


labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
for i in np.arange(6):
    axs[i].set_title(labels[i], loc='left', fontsize=10)

plt.savefig('Fig4.pdf', dpi=300)