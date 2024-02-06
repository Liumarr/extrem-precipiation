import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker

def vor(u, v):
    dx                     = 500
    ugrad                  = np.array(np.gradient(u))
    vgrad                  = np.array(np.gradient(v))
    vor                    = (vgrad[1,:,:]-ugrad[0,:,:])/dx
    return vor



data            = xr.open_dataset("../data/vor.nc")

acp                    = np.array(data.acp)
u05                    = np.array(data.u05)
v05                    = np.array(data.v05)
u1                     = np.array(data.u1)
v1                     = np.array(data.v1)
u3                     = np.array(data.u3)
v3                     = np.array(data.v3)

vws1                   = np.array(data.vws1)
vws3                   = np.array(data.vws3)
vor05                  = np.zeros((884,161,161))
vor1                   = vor05
for i in np.arange(884):
    vor05[i,:,:] = vor(u05[i,:,:], v05[i,:,:])
    vor1[i,:,:]  = vor(u1[i,:,:], v1[i,:,:])
fn            = '../avgvor-mvs.csv'
vor           = np.array(pd.read_csv(fn))


mv = []
nonmv = []
print(vor.shape)
for i in np.arange(884):
    if vor[i,0] > 0.0001:
        mv.append(i)
    else:
        nonmv.append(i)
print(len(mv))
acp_all        = np.mean(acp, axis=0)
acp_ex         = np.mean(acp[nonmv,:,:], axis=0)
acp_mv         = np.mean(acp[mv,:,:], axis=0)
vws1_all       = np.mean(vws1, axis=0)
vws1_ex        = np.mean(vws1[nonmv,:,:], axis=0)
vws1_mv        = np.mean(vws1[mv,:,:], axis=0)
vws3_all       = np.mean(vws3, axis=0)
vws3_ex        = np.mean(vws3[nonmv,:,:], axis=0)
vws3_mv        = np.mean(vws3[mv,:,:], axis=0)

u05_all        = np.mean(u05, axis=0)
u05_ex         = np.mean(u05[nonmv,:,:], axis=0)
u05_mv         = np.mean(u05[mv,:,:], axis=0)

print(u05_all)
v05_all        = np.mean(v05, axis=0)
v05_ex         = np.mean(v05[nonmv,:,:], axis=0)
v05_mv         = np.mean(v05[mv,:,:], axis=0)

u1_all        = np.mean(u1, axis=0)
u1_ex         = np.mean(u1[nonmv,:,:], axis=0)
u1_mv         = np.mean(u1[mv,:,:], axis=0)

v1_all        = np.mean(v1, axis=0)
v1_ex         = np.mean(v1[nonmv,:,:], axis=0)
v1_mv         = np.mean(v1[mv,:,:], axis=0)

u3_all        = np.mean(u3, axis=0)
u3_ex         = np.mean(u3[nonmv,:,:], axis=0)
u3_mv         = np.mean(u3[mv,:,:], axis=0)

v3_all        = np.mean(v3, axis=0)
v3_ex         = np.mean(v3[nonmv,:,:], axis=0)
v3_mv         = np.mean(v3[mv,:,:], axis=0)


# v05                    = np.array(data.v05)
# u1                     = np.array(data.u1)
# v1                     = np.array(data.v1)
# u3                     = np.array(data.u3)s
# v3                     = np.array(data.v3)





vor05_all      = 10**4*np.mean(vor05, axis=0)
vor05_ex       = 10**4*np.mean(vor05[nonmv,:,:], axis=0)
vor05_mv       = 10**4*np.mean(vor05[mv,:,:], axis=0)
vor1_ex        = 10**4*np.mean(vor1[nonmv,:,:], axis=0)
vor1_mv        = 10**4*np.mean(vor1[mv,:,:], axis=0)
vor1_all       = 10**4*np.mean(vor1, axis=0)

plt.rc('font',family='Times New Roman')
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.subplots_adjust(hspace=0.4, wspace=0.1)

fig, ax       = plt.subplots(4, 3, sharex=True, sharey=True,\
                             subplot_kw=dict(aspect='equal'))
axs           = ax.flatten()

yl            = MultipleLocator(20)
x             = np.arange(-40,40.5,0.5)
y             = np.arange(-40,40.5,0.5)
levels         = np.array([np.arange(0,32,2),np.arange(-3,3.4,0.4),np.arange(0,16,1),np.arange(0,16,1)])

axs[0].yaxis.set_major_locator(yl)
axs[0].xaxis.set_major_locator(yl)
tick           = [[0,10,20,30],[-3,-1,1,3],[0,5,10,15],[0,5,10,15]]
k=0
j=0
for i in [acp_all, acp_ex, acp_mv, vor05_all, vor05_ex, vor05_mv, vws1_all, vws1_ex, vws1_mv, vws3_all, vws3_ex, vws3_mv]:
    p = axs[k].contourf(x, y, i, levels=levels[int(j),:], cmap='rainbow',alpha=0.9)
    axs[k].tick_params(which='major',length=2)

    if k==2 or k==5 or k==8 or k==11:
        cb= fig.colorbar(p, ax=axs[k], shrink=0.95)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb.locator = tick_locator
        cb.ax.tick_params(length=1.6, width=0.8) 
        cb.set_ticks(tick[j])
        cb.update_ticks()
        j=j+1
    k=k+1




wx = np.linspace(-40, 40, 11)

wX, wY = np.meshgrid(wx, wx)
print(wX.shape)

# axs[0].barbs(wX,wY,u05_all[::16,::16],v05_all[::16,::16],length=4,pivot='middle',linewidth=0.5)
# axs[1].barbs(wX,wY,u05_ex[::16,::16],v05_ex[::16,::16],length=4,pivot='middle',linewidth=0.5)
# axs[2].barbs(wX,wY,u05_mv[::16,::16],v05_mv[::16,::16],length=4,pivot='middle',linewidth=0.5)

# axs[3].barbs(wX,wY,u1_all[::16,::16],v1_all[::16,::16],length=4,pivot='middle',linewidth=0.5)
# axs[4].barbs(wX,wY,u1_ex[::16,::16],v1_ex[::16,::16],length=4,pivot='middle',linewidth=0.5)
# axs[5].barbs(wX,wY,u1_mv[::16,::16],v1_mv[::16,::16],length=4,pivot='middle',linewidth=0.5)

# axs[6].barbs(wX,wY,u3_all[::16,::16],v3_all[::16,::16],length=4,pivot='middle',linewidth=0.5)
# axs[7].barbs(wX,wY,u3_ex[::16,::16],v3_ex[::16,::16],length=4,pivot='middle',linewidth=0.5)
# axs[8].barbs(wX,wY,u3_mv[::16,::16],v3_mv[::16,::16],length=4,pivot='middle',linewidth=0.5)

labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']
for i in np.arange(12):
    axs[i].set_title(labels[i], loc='left', fontsize=8)


axs[0].set_ylabel('Y (km)', fontsize=6)
axs[3].set_ylabel('Y (km)', fontsize=6)
axs[6].set_ylabel('Y (km)', fontsize=6)
axs[9].set_ylabel('Y (km)', fontsize=6)
axs[9].set_xlabel('X (km)', fontsize=6)
axs[10].set_xlabel('X (km)', fontsize=6)
axs[11].set_xlabel('X (km)', fontsize=6)



plt.savefig('Fig3-1215.pdf')

