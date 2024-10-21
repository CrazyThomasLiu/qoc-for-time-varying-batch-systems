import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
save_figure=False
batch_length = 100
batch_explore=30
n=3 # state
m=2 # input
q=2 # output
"1. Load the output response from csv"

df_y_out = pd.read_csv('./Data/Buffer_y_{}.csv'.format(batch_explore), index_col=0)
y_out = df_y_out.to_numpy()
y_out = y_out.reshape(q, int(y_out.shape[0] / q), batch_explore)
"2. the reference trajectory"

y_ref = np.ones((batch_length+1, q))
y_ref[0:51,0]=10.* y_ref[0:51,0]
y_ref[0:51,1]=20.* y_ref[0:51,1]
y_ref[51:101,0]=20.* y_ref[51:101,0]
y_ref[51:101,1]=30.* y_ref[51:101,1]


pdb.set_trace()

"3. plot the 3D figure for the exploration"
plt.rcParams['pdf.fonttype'] = 42
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.invert_xaxis()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
batch_axis=np.ones(batch_length+1,dtype=int)
t=range(batch_length+1)

ax.plot3D(batch_axis*(batch_explore),t, y_ref[:,0].squeeze(),linestyle='dashed',linewidth=1.5,color='#97b319')
for batch in range(batch_explore):
    batch_plot=batch_axis*(batch+1)
    ax.plot3D(batch_plot, t, y_out[0,:,batch].squeeze(), linewidth=1.5, color='black')
xlable = 'Batch:$k$'
ylable = 'Time:$t$'
zlable = 'Output Response'

font3 = {'family': 'Arial',
         'weight': 'bold',
         'size': 15
         }

ax.set_xlabel(xlable,font3)
ax.set_ylabel(ylable,font3)
ax.set_zlabel(zlable,font3)
ax.legend(['$y_{k,t}^{r}$','$y_{k,t}$'],fontsize=13)
plt.tick_params(labelsize=12)
ax.view_init(40, -19)



if save_figure==True:
    plt.savefig('sample_data_3D_MIMO.jpg',dpi=800)
    #plt.savefig('sample_data_3D_MIMO.pdf')
plt.show()

