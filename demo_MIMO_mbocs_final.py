import copy
import pdb
import control
from algorithm.model_based_ocs import MBOCS
from env.time_variant_batch_sys import Time_varying_batch_system
import numpy as np
import matplotlib.pyplot as plt
# save the figure or not

save_figure=False
save_csv=False
batch_length = 100
batch_num=50
# cost function
Q = np.matrix([[200.,0.],[0.,100.]])
R = np.matrix([[0.01,0.],[0.,0.01]])
# the time-varying weighting matrix
Q_t = []
R_t = []
for time in range(batch_length):
    Q_t.append(copy.deepcopy(Q))
    R_t.append(copy.deepcopy(R))

"1. the real state space"
'1.1 the time-invariant parameters'
A= np.matrix([[0., 1.0,-1.0], [-0.6, -1.1,0.6],[-0.4, -1.2,0.5]])
B= np.matrix([[1.,2.], [10.,6.], [5.,3.]])
C= np.matrix([[1.0,1.5,1.],[1.0,0.5,0.5]])
n=3 # state
m=2 # input
q=2 # output
'1.2 the time-varying parameters'
A_t = []
B_t = []
C_t=  []
for time in range(batch_length):
    A_t.append(copy.deepcopy(A))
    B_t.append(copy.deepcopy(B))
    C_t.append(copy.deepcopy(C))
C_t.append(copy.deepcopy(C))
"1.3 add the system uncertainty"
A_t_env = []
B_t_env = []
C_t_env=  []
for time in range(batch_length):
    A_t_env.append(copy.deepcopy(A))
    A_t_env[time]=A_t[time]+A_t[time]*0.1*np.sin(time)
    B_t_env.append(copy.deepcopy(B))
    B_t_env[time] =B_t[time]+B_t[time] *0.2* np.cos(time)
    C_t_env.append(copy.deepcopy(C))
    C_t_env[time] = C_t[time] + C_t[time] * 0.05 * np.sin(2*time)
'1.3 add a additional parameter for the t+1'
C_t_env.append(copy.deepcopy(C))
C_t_env[batch_length] = C_t[batch_length] + C_t[batch_length] * 0.05 * np.sin(2*batch_length)

'1.4 the reference trajectory'

y_ref = np.ones((batch_length+1, q))
y_ref[0:51,0]=10.* y_ref[0:51,0]
y_ref[0:51,1]=20.* y_ref[0:51,1]
y_ref[51:101,0]=20.* y_ref[51:101,0]
y_ref[51:101,1]=30.* y_ref[51:101,1]

'1.5 calculate the reference trajectory model D_{t},H_{t}'
y_sum = np.zeros((batch_length+1, 1))

for time in range(batch_length+1):
    for output_dim in range(q):
        y_sum[time,0]+=y_ref[time,output_dim]

D_t= []
for time in range(batch_length):
    D_t.append(np.zeros((1,1)))
    D_t[time][0, 0] = y_sum[time + 1, 0] / y_sum[time, 0]
H_t= []
for time in range(batch_length+1):
    H_t.append(np.zeros((q,1)))
    for output_dim in range(q):
        H_t[time][output_dim, 0] = y_ref[time, output_dim] / y_sum[time, 0]
"2. compute the model-based optimal control law"
'2.1 initial the model-based control scheme'
'model-based optimal control policy without the time-varying system uncertainties'
mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t,H_t=H_t,Q_t=Q_t, R_t=R_t)
'model-based optimal control policy with the time-varying system uncertainties'
full_ocs = MBOCS(batch_length=batch_length, A_t=A_t_env, B_t=B_t_env, C_t=C_t_env,D_t=D_t,H_t=H_t,Q_t=Q_t, R_t=R_t)
mb_ocs.control_law()
full_ocs.control_law()
"2.2 save the model-based control policy and the optimal control policy"
mb_ocs.save_K(name='MIMO/initial_control_law')
full_ocs.save_K(name='MIMO/optimal_control_law')
