import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

save_figure=True
batch_length = 200
batch_explore=10
n=2 # state
m=1 # input
q=1 # output
"1. Load the output response from csv"

df_y_out = pd.read_csv('./Data/Injection_Molding/Buffer_y_{}.csv'.format(batch_explore), index_col=0)
y_out = df_y_out.to_numpy()
y_out = y_out.reshape(q, int(y_out.shape[0] / q), batch_explore)
"2. the reference trajectory"

y_ref = np.ones((batch_length+1, q))
y_ref[0]=200.* y_ref[0]
y_ref[1:101]=200.* y_ref[1:101]
for time in range(101,121):
    y_ref[time,0]=200.+5*(time-100.)
y_ref[121:] = 300. * y_ref[121:]


"3. plot the 3D figure for the exploration"
plt.rcParams['pdf.fonttype'] = 42
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
batch_axis=np.ones(batch_length+1,dtype=int)
t=range(batch_length+1)

ax.plot3D(batch_axis*(batch_explore),t, y_ref[:,0].squeeze(),linestyle='dashed',linewidth=2,color='#97b319')
for batch in range(batch_explore):
    batch_plot=batch_axis*(batch+1)
    ax.plot3D(batch_plot, t, y_out[0,:,batch].squeeze(), linewidth=1.5, color='black')
xlable = 'Batch:$k$'
ylable = 'Time:$t$'
zlable = 'Output Response'

font3 = {'family': 'Arial',
         'weight': 'bold',
         'size': 13
         }

ax.set_xlabel(xlable,font3)
ax.set_ylabel(ylable,font3)
ax.set_zlabel(zlable,font3)
ax.legend(['$y_{r,t}$','$y_{k,t}$'],fontsize=12)
ax.tick_params(labelsize=11)
ax.view_init(40, -19)



if save_figure==True:
    plt.savefig('sample_data_3D_injection_molding_final.pdf')
plt.show()

