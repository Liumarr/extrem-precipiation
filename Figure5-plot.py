import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker

def MFC(u, v, q):
    # MFC = divg(uq,uv) = -(u*dq/dx+v*dq/dy)-q*((du/dx)+(dv/dy))
    dx                     = 500
    ugrad                  = np.array(np.gradient(u))/dx
    vgrad                  = np.array(np.gradient(v))/dx
    qgrad                  = np.array(np.gradient(q))/dx
    result                 = -1000*(u*qgrad[0,:,:]+v*qgrad[1,:,:])-q*((ugrad[0,:,:])+vgrad[1,:,:])
    return result 

data                     = xr.open_dataset("uvq_3000m.nc")


fn            = '../avgvor-mvs.csv'
vor           = np.array(pd.read_csv(fn))
mv    = []
nonmv = []
mfcc   = []
for i in np.arange(884):
    mfcc.append(MFC(data.u[i,:,:],  data.v[i,:,:], data.q[i,:,:]))

    if vor[i,0] > 0.0001:
        mv.append(i)
    else:
        nonmv.append(i)

print(len(mv))
mfcc = np.array(mfcc)

mfcc_all        = np.mean(mfcc, axis=0)
mfcc_ex         = np.mean(mfcc[nonmv,:,:], axis=0)
mfcc_mv         = np.mean(mfcc[mv,:,:], axis=0)

u_all           = np.mean(data.u, axis=0)
u_ex            = np.mean(data.u[nonmv,:,:], axis=0)
u_mv            = np.mean(data.u[mv,:,:], axis=0)

v_all           = np.mean(data.v, axis=0)
v_ex            = np.mean(data.v[nonmv,:,:], axis=0)
v_mv            = np.mean(data.v[mv,:,:], axis=0)


plt.rc('font',family='Times New Roman')
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig, ax       = plt.subplots(1, 2, sharex=True, sharey=True, \
                             subplot_kw=dict(aspect='equal'))
axs           = ax.flatten()

yl            = MultipleLocator(20)
x             = np.arange(-40,40.5,0.5)
y             = np.arange(-40,40.5,0.5)

axs[0].yaxis.set_major_locator(yl)
axs[0].xaxis.set_major_locator(yl)
tick           = [ -0.5, -0.25, 0, 0.25, 0.5]
k=0
j=0
for i in [mfcc_ex, mfcc_mv]:
    p = axs[k].contourf(x, y, i, levels=np.arange(-0.5,0.52,0.02), cmap='rainbow',alpha=0.9)
    axs[k].tick_params(which='major',length=2)

 
    cb= fig.colorbar(p, ax=axs[k], shrink=0.5)
    # tick_locator = ticker.MaxNLocator(nbins=4)
    #cb.locator = tick_locator
    cb.ax.tick_params(length=1.6, width=0.8) 
    cb.set_ticks(tick)
    # cb.update_ticks()

    k=k+1

wx = np.linspace(-40, 40, 11)
wX, wY = np.meshgrid(wx, wx)
axs[0].barbs(wX,wY,u_ex[::16,::16],v_ex[::16,::16],length=4,pivot='middle',linewidth=0.5,barb_increments=dict(half=2, full=4, flag=20))
axs[1].barbs(wX,wY,u_mv[::16,::16],v_mv[::16,::16],length=4,pivot='middle',linewidth=0.5,barb_increments=dict(half=2, full=4, flag=20))

axs[0].set_ylabel('Y (km)', fontsize=10)
axs[0].set_xlabel('X (km)', fontsize=10)
axs[1].set_ylabel('Y (km)', fontsize=10)
axs[1].set_xlabel('X (km)', fontsize=10)
axs[0].set_title('(a)', loc='left', fontsize=10)
axs[1].set_title('(b)', loc='left', fontsize=10)



plt.savefig('Fig5-1224-3km.pdf')