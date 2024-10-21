import copy
import pdb
import control
from algorithm.model_based_ocs import MBOCS
from env.time_variant_batch_sys import Time_varying_batch_system
from algorithm.q_learning_ocs import Q_learning_OCS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# save the figure or not

save_figure=True
save_csv=False
batch_length = 200
batch_explore=10
# cost function
Q = np.matrix([[100.]])
R = np.matrix([[0.1]])
# the time-varying weighting matrix
Q_t = []
R_t = []
for time in range(batch_length):
    Q_t.append(copy.deepcopy(Q))
    R_t.append(copy.deepcopy(R))

"1. the real state space"
'1.1 the time-invariant parameters'
A= np.matrix([[1.607, 1.0], [-0.6086, 0.0]])
B= np.matrix([[1.239], [-0.9282]])
C= np.matrix([[1.0,0.0]])
n=2 # state
m=1 # input
q=1 # output
'1.2 the time-varying parameters'
A_t = []
B_t = []
C_t=  []
for time in range(batch_length):
    A_t.append(copy.deepcopy(A))
    A_t[time]=A_t[time]*(0.5+0.2*np.exp(time/200))
    B_t.append(copy.deepcopy(B))
    B_t[time] = B_t[time] *(1+0.1*np.exp(time/200))
    C_t.append(copy.deepcopy(C))
C_t.append(copy.deepcopy(C))
"1.3 add the system uncertainty"
A_t_env = []
B_t_env = []
C_t_env=  []
delta_A_t=np.matrix([[0.0604, -0.0204], [-0.0204, 0.0]])
delta_B_t=np.matrix([[0.062], [-0.0464]])
delta_C_t=np.matrix([[0.01,-0.01]])
for time in range(batch_length):
    A_t_env.append(copy.deepcopy(A))
    A_t_env[time] = A_t[time] + delta_A_t * 1.0*np.exp(time/200)
    B_t_env.append(copy.deepcopy(B))
    B_t_env[time] = B_t[time]+delta_B_t*np.sin(time)
    C_t_env.append(copy.deepcopy(C))
    C_t_env[time] = C_t[time] + delta_C_t *np.sin(time)
C_t_env.append(copy.deepcopy(C))
C_t_env[batch_length] = C_t[batch_length] + delta_C_t *np.sin(batch_length)

'1.4 the reference trajectory'

y_ref = np.ones((batch_length+1, q))
y_ref[0]=200.* y_ref[0]
y_ref[1:101]=200.* y_ref[1:101]
for time in range(101,121):
    y_ref[time,0]=200.+5*(time-100.)
y_ref[121:] = 300. * y_ref[121:]


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
mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t,H_t=H_t,Q_t=Q_t, R_t=R_t)
mb_ocs.load_K(name='Injection_Molding/initial_control_law')

"3. set the simulated env"
'3.1 the initial state'
x_k0 = np.array([[50.], [50.]])
sample_time = 1

"3.2 set the batch system"
def state_update(t, x, u, params):
    # get the parameter from the params
    # Map the states into local variable names
    # the state x_{k,t}
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # the control signal u_{k,t}
    u1 = np.array([u[0]])
    # Get the current time t state space
    A_t_env=params.get('A_t')
    B_t_env = params.get('B_t')
    # Compute the discrete updates
    dz1 = A_t_env[0,0]* z1 + A_t_env[0,1]* z2 +B_t_env[0,0]*u1
    dz2 = A_t_env[1,0]* z1 + A_t_env[1,1]* z2 +B_t_env[1,0]*u1
    return [dz1, dz2]


def output_update(t, x, u, params):
    # Parameter setup
    # Compute the discrete updates
    # the state x_{k,t}
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # Get the current time t state space
    C_t_env=params.get('C_t')
    y1 = C_t_env[0,0]* z1 + C_t_env[0,1]* z2
    return [y1]

batch_sys = control.NonlinearIOSystem(
    state_update, output_update, inputs=('u1'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=1, name='Injection_molding')

controlled_system = Time_varying_batch_system(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t_env, B_t=B_t_env, C_t=C_t_env)



"4. initial the q-learning control scheme"
Q_learning=Q_learning_OCS(batch_length=batch_length,D_t=D_t,H_t=H_t,n=n,m=m,q=q,Q_t=Q_t,R_t=R_t)
Q_learning.load_K(path='Data/Injection_Molding/policy_iteration_10/q_learning_control_policy15.csv')

"5. simulation model-based optimal control scheme"

"5.1 reset the system"
x_tem,y_out=controlled_system.reset()

"5.2 simulations"
mb_ocs_y=np.zeros((q,batch_length+1))
mb_ocs_u=np.zeros((m,batch_length))
mb_ocs_y[:,0:1]=y_out[:,0]   # the matrix and array
for time in range(batch_length):
    state = np.block([[x_tem],
                      [y_sum[time + 1]]])
    control_signal = -mb_ocs.K[time]@state
    x_tem,y_out= controlled_system.step(control_signal)
    mb_ocs_y[:,time+1] = y_out[:, 0]
    mb_ocs_u[:, time:time+1] = control_signal[:, 0]

"6. simulation Q-learning "

"6.1 reset the system"
x_tem,y_out=controlled_system.reset()

"6.2 simulations"
Q_learning_y=np.zeros((q,batch_length+1))
Q_learning_u=np.zeros((m,batch_length))
Q_learning_y[:,0:1]=y_out[:,0]   # the matrix and array
for time in range(batch_length):
    state = np.block([[x_tem],
                      [y_sum[time + 1]]])
    control_signal = -Q_learning.K[time]@state
    x_tem,y_out= controlled_system.step(control_signal)
    Q_learning_y[:,time+1] = y_out[:, 0]
    Q_learning_u[:, time:time+1] = control_signal[:, 0]


"7. compute the RMSE and the SAE"
df_y_robust = pd.read_csv('./PI_Robust_output_final.csv', index_col=0)
y_robust = df_y_robust.to_numpy()

df_u_robust = pd.read_csv('./PI_Robust_control_final.csv', index_col=0)
u_robust = df_u_robust.to_numpy()
RMSE_mb = 0

for time in range(batch_length):
    RMSE_mb += (np.linalg.norm(y_ref[time + 1, :]-mb_ocs_y[:,time + 1] )) ** 2
    #pdb.set_trace()
RMSE_mb = math.sqrt(RMSE_mb / batch_length)

print("RMSE_mb:",RMSE_mb)

RMSE_Q_learning = 0


for time in range(batch_length):
    RMSE_Q_learning += (np.linalg.norm(y_ref[time + 1, :]-Q_learning_y[:,time + 1] )) ** 2
    #pdb.set_trace()
RMSE_Q_learning = math.sqrt(RMSE_Q_learning / batch_length)

print("RMSE_Q_learning:",RMSE_Q_learning)
RMSE_robust = 0


for time in range(batch_length):
    RMSE_robust += (np.linalg.norm(y_ref[time + 1, :]-y_robust[time + 1,:] )) ** 2
    #pdb.set_trace()
RMSE_robust = math.sqrt(RMSE_robust / batch_length)


print("RMSE_robust:",RMSE_robust)



SAE_mb = 0


for time in range(batch_length):
    SAE_mb += abs(y_ref[time + 1, :]-mb_ocs_y[:,time + 1] )


print("SAE_mb:",SAE_mb)

SAE_Q_learning = 0


for time in range(batch_length):
    SAE_Q_learning += abs(y_ref[time + 1, :]-Q_learning_y[:,time + 1] )


print("SAE_Q_learning:",SAE_Q_learning)


SAE_robust = 0


for time in range(batch_length):
    SAE_robust += abs(y_ref[time + 1, :]-y_robust[time + 1,:] )

print("SAE_robust:",SAE_robust)



"8. plot the figures"
plt.rcParams['pdf.fonttype'] = 42


fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, constrained_layout=True,figsize=((10,6)))
#fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True,figsize=((10,6)))
"8.1 plot the output response"

time=range(0,batch_length+1)
ax0.plot(time, y_ref[0:,0],linewidth=1.5,color='#46788e', label='smoothed')
ax0.plot(time, mb_ocs_y[0],linewidth=1.5,color='orange')
ax0.plot(time, Q_learning_y[0],linewidth=1.5,color='black',linestyle='dotted')
ax0.plot(time, y_robust,color='green',linewidth=1.5)

ax0.grid()
#plt.ylim((100,350))
xlable = 'Time:$t$'
ylable = 'Output'
font = {'family': 'Arial',
         'weight': 'bold',
         'size': 18,
         }
ax0.set_ylabel('Output:$y^{1}$',font)
ax0.tick_params(labelsize=14)
ax0.legend(['Reference Trajectory','Model-based Initial Control Policy','Q-learning-based Optimal Control Policy','PI-based Indirect ILC'],fontsize=12)




"7.2 plot the control signal"

time=range(1,batch_length+1)
ax1.plot(time, mb_ocs_u[0],linewidth=2,color='orange')
ax1.plot(time, Q_learning_u[0],linewidth=2,color='black',linestyle='dotted')
ax1.plot(time, u_robust,color='green',linewidth=1.5)
ax1.grid()
xlable = 'Time:$t$'
ylable = 'Control Signal'
ax1.set_ylabel('Control Signal:$u^{1}$',font)
ax1.set_xlabel('Time:$t$',font)
ax1.tick_params(labelsize=14)



if save_figure==True:
    plt.savefig('Q_learning_injection_molding_final.pdf')

plt.show()

