import copy
import pdb
import random
import control
import os
import sys
from algorithm.model_based_ocs import MBOCS
from env.time_variant_batch_sys import Time_varying_batch_system
import numpy as np
import matplotlib.pyplot as plt
# save the figure or not

save_figure=False
save_csv=False
#optimal_batch=10
#dis_fac=0.95
batch_length = 200
batch_num=50
# cost function
Q = np.matrix([[100.]])
R = np.matrix([[0.1]])
# the time-varying weighting matrix
Q_t = []
R_t = []
for time in range(batch_length):
    Q_t.append(copy.deepcopy(Q))
    R_t.append(copy.deepcopy(R))

#pdb.set_trace()
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

#pdb.set_trace()
'1.4 the reference trajectory'

y_ref = np.ones((batch_length+1, q))
y_ref[0]=200.* y_ref[0]
y_ref[1:101]=200.* y_ref[1:101]
for time in range(101,121):
    y_ref[time,0]=200.+5*(time-100.)
y_ref[121:] = 300. * y_ref[121:]

#pdb.set_trace()

'1.5 calculate the reference trajectory model D_{t},H_{t}'
y_sum = np.zeros((batch_length+1, 1))

for time in range(batch_length+1):
    for output_dim in range(q):
        y_sum[time,0]+=y_ref[time,output_dim]

D_t= []
for time in range(batch_length):
    D_t.append(np.zeros((1,1)))
    #pdb.set_trace()
    D_t[time][0, 0] = y_sum[time + 1, 0] / y_sum[time, 0]
H_t= []
for time in range(batch_length+1):
    H_t.append(np.zeros((q,1)))
    for output_dim in range(q):
    #pdb.set_trace()
        H_t[time][output_dim, 0] = y_ref[time, output_dim] / y_sum[time, 0]
#pdb.set_trace()
"2. compute the model-based optimal control law"
#mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t_env, B_t=B_t_env, C_t=C_t_env,D_t=D_t,H_t=H_t,Q_t=Q_t, R_t=R_t)
mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t,H_t=H_t,Q_t=Q_t, R_t=R_t)
#pdb.set_trace()
mb_ocs.control_law()
"save the initial control policy"
#pdb.set_trace()
#mb_ocs.save_K(name='Injection_Molding/optimal_control_law')
#mb_ocs.save_K(name='Injection_Molding/initial_control_law')
#pdb.set_trace()
"3. set the simulated env"
'3.1 the initial state'
x_k0 = np.array([[50.], [50.]])
sample_time = 1

#pdb.set_trace()
"3.2 set the batch system"
def state_update(t, x, u, params):
    # get the parameter from the params
    # pdb.set_trace()
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
    # pdb.set_trace()
    return [dz1, dz2]


def output_update(t, x, u, params):
    # Parameter setup
    # pdb.set_trace()
    # Compute the discrete updates
    # the state x_{k,t}
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # Get the current time t state space
    C_t_env=params.get('C_t')
    #pdb.set_trace()
    y1 = C_t_env[0,0]* z1 + C_t_env[0,1]* z2
    return [y1]

batch_sys = control.NonlinearIOSystem(
    state_update, output_update, inputs=('u1'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=1, name='Injection_molding')

controlled_system = Time_varying_batch_system(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t_env, B_t=B_t_env, C_t=C_t_env)




"4. simulations"

"4.1 reset the system"
x_tem,y_out=controlled_system.reset()

#pdb.set_trace()


"4.2 simulations"
fig_y=np.zeros((q,batch_length+1))
fig_u=np.zeros((m,batch_length))
fig_y[:,0:1]=y_out[:,0]   # the matrix and array
for time in range(batch_length):
    state = np.block([[x_tem],
                      [y_sum[time + 1]]])
    control_signal = -mb_ocs.K[time]@state
    x_tem,y_out= controlled_system.step(control_signal)
    #pdb.set_trace()
    fig_y[:,time+1] = y_out[:, 0]
    fig_u[:, time:time+1] = control_signal[:, 0]


#pdb.set_trace()
"5. plot the figures"
"set the global parameters"
config = {
    "font.family": 'sans-serif',
    "font.serif": ['Arial'],
    "font.size": 12,
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
#pdb.set_trace()
"5.1 plot the output response"
"output 1"
time=range(0,batch_length+1)
ax0.plot(time, y_ref[0:,0], label='smoothed')
ax0.plot(time, fig_y[0],linestyle='dotted')


ax0.grid()
xlable = 'Time:$t$'
ylable = 'Output:$y^{1}$'
font = {'family': 'Arial',
         'weight': 'bold',
         'size': 16,
         }
ax0.set_ylabel('Output',font)
#plt.xlabel(xlable,font)
#plt.ylabel(ylable,font)
#plt.legend(['Reference Trajectory','Output Response'],loc='upper right')
#plt.legend(['Reference Trajectory','Output Response'])
ax0.legend(['Reference Trajectory','Q-learning-based Optimal Controller'])
#plt.show()
#pdb.set_trace()

"5.2 plot the control signal"
"control signal 1"
#pdb.set_trace()

time=range(1,batch_length+1)
ax1.plot(time, fig_u[0], label='smoothed')

ax1.grid()
xlable = 'Time:$t$'
ylable = 'Control Signal'
ax1.set_ylabel('Control Signal',font)
ax1.set_xlabel('Time:$t$',font)
#plt.legend(['Reference Trajectory','Output Response'],loc='upper right')
#plt.legend(['Reference Trajectory','Output Response'])


if save_figure==True:
    plt.savefig('mb_ocs_injection_molding.pdf')
plt.show()

pdb.set_trace()
a=2