import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


plt.rc('font',family='Times New Roman')
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

data0   = pd.read_csv('MVs.csv')
fig, _axs       = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
axs             = _axs.flatten()
fig.subplots_adjust(hspace=0.3, wspace=0.3)


#plt.grid()
interval=stats.norm.interval(0.95,np.mean(data0.rain),np.std(data0.rain))
print(interval)

sns.violinplot(ax=axs[0], x='MVs', y='rain', hue="MVs",palette="Set2",
               data=data0,split=True,inner='quartiles',cut=0,legend=True)
axs[0].set_ylim([20,60])
axs[0].set_ylabel('Rain Rate (mm / 10mins)')
axs[0].axhline(y=interval[1],c='grey',ls='--',lw=1)
axs[0].axhline(y=interval[0],c='grey',ls='--',lw=1)

interval=stats.norm.interval(0.95,np.mean(data0.vr500),np.std(data0.vr500))
sns.violinplot(ax=axs[1], x='MVs', y='vr500', hue="MVs",palette="Set2",
               data=data0,split=True,inner='quartiles',cut=0)
axs[1].set_ylim([0,10])
axs[1].set_yticks([0, 2, 4, 6, 8, 10])
axs[1].set_ylabel(r'$\mathrm{Vorticity_{500m}\/(10^{-3} s^{-1})}$')
# axs[1].axhline(y=interval[1],c='grey',ls='--',lw=1)
# axs[1].axhline(y=interval[0],c='grey',ls='--',lw=1)



sns.violinplot(ax=axs[2], x='MVs', y='MEV', hue="MVs",palette="Set2",
               data=data0,split=True,inner='quartiles',cut=0, legend=False)
axs[2].set_ylim([0,8])
axs[2].set_yticks([0, 2, 4, 6, 8])
axs[2].set_ylabel('MEV '+r'$\mathrm{(10^{-4} s^{-1})}$')
# interval=stats.norm.interval(0.95,np.mean(data0.MEV),np.std(data0.MEV))
# axs[2].axhline(y=interval[1],c='grey',ls='--',lw=1)
# axs[2].axhline(y=interval[0],c='grey',ls='--',lw=1)


sns.violinplot(ax=axs[3], x='MVs', y='HMV', hue="MVs",palette="Set2",
               data=data0,split=True,inner='quartiles',cut=0, legend=False)
axs[3].set_ylim([0,4000])
axs[3].set_yticks([0, 1000, 2000, 3000, 4000])
axs[3].set_yticklabels(['0', '1', '2', '3','4'])
axs[3].set_ylabel('HMEV (km)')
# interval=stats.norm.interval(0.95,np.mean(data0.HMV),np.std(data0.HMV))
# axs[3].axhline(y=interval[1],c='grey',ls='--',lw=1)
# axs[3].axhline(y=interval[0],c='grey',ls='--',lw=1)


sns.violinplot(ax=axs[4], x='MVs', y='vws1', hue="MVs",palette="Set2",
               data=data0,split=True,inner='quartiles',cut=0, legend=False)
axs[4].set_ylim([0,20])
axs[4].set_yticks([0, 5, 10, 15, 20])
axs[4].set_ylabel(r'$\mathrm{\overline{VWS}_{0-1km}\/(m s^{-1})}$')
# interval=stats.norm.interval(0.95,np.mean(data0.vws1),np.std(data0.vws1))
# axs[4].axhline(y=interval[1],c='grey',ls='--',lw=1)
# axs[4].axhline(y=interval[0],c='grey',ls='--',lw=1)

sns.violinplot(ax=axs[5], x='MVs', y='vws3', hue="MVs",palette="Set2",
               data=data0,split=True,inner='quartiles',cut=0, legend=False)
axs[5].set_ylim([0,25])
axs[5].set_yticks([0, 5, 10, 15, 20, 25])
axs[5].set_ylabel(r'$\mathrm{\overline{VWS}_{0-3km}\/(m s^{-1})}$')
# interval=stats.norm.interval(0.95,np.mean(data0.vws3),np.std(data0.vws3))
# axs[5].axhline(y=interval[1],c='grey',ls='--',lw=1)
# axs[5].axhline(y=interval[0],c='grey',ls='--',lw=1)


labels = ['(a)','(b)','(c)','(d)','(e)','(f)']
for i in np.arange(6):
    axs[i].set_title(labels[i], loc='left', fontsize=14)


plt.savefig('Figure-violin-1219.pdf')
# output.to_csv('MVs1.csv')