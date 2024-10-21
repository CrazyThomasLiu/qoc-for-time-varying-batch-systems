import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MaxNLocator
import pandas as pd
import math
save_figure=False
batch_length = 100
batch_explore=25
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

"3. the reference trajectory"
RMSE = np.zeros(batch_explore)

for batch in range(batch_explore):
    for time in range(batch_length):
        RMSE[batch] += (np.linalg.norm(y_ref[time + 1, :]-y_out[:,time + 1,batch] )) ** 2
        #pdb.set_trace()
    RMSE[batch] = math.sqrt(RMSE[batch] / batch_length)
#pdb.set_trace()

'4 Plot the RMSE'
plt.rcParams['pdf.fonttype'] = 42
batch_axis=range(1,batch_explore+1)
fig,ax0=plt.subplots(1,1)
x_major_locator=MultipleLocator(int(batch_explore/10))
ax0=plt.gca()
ax0.xaxis.set_major_locator(x_major_locator)
ax0.plot(batch_axis,RMSE,linewidth=2,color='#46788e',linestyle = 'dashdot')
#plt.plot(batch_axis,RMSE_PI,linewidth=1.5,color='orange',linestyle='solid')
ax0.grid()

xlable = 'Batch:$k$'
ylable = 'RMSE'
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 15,
         }
plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
plt.tick_params(labelsize=12)
plt.xlim((0,25))
#ax0.legend(['Q-learning-based Optimal Controller','PI-based Indirect ILC [JPC,2019]'],fontsize=11)



if save_figure==True:
    #plt.savefig('sample_RMSE_MIMO.jpg',dpi=800)
    plt.savefig('sample_RMSE_MIMO.pdf')
plt.show()

