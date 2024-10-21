import copy
import pdb
import random
import control
import os
import sys
from algorithm.model_based_ocs import MBOCS
from env.time_variant_batch_sys import Time_varying_batch_system
from algorithm.q_learning_ocs import Q_learning_OCS
import numpy as np
import matplotlib.pyplot as plt
# save the figure or not

save_figure=False
save_csv=False
#optimal_batch=10
#dis_fac=0.95
batch_length = 100
batch_explore=25
#batch_num=50
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
#pdb.set_trace()
C_t.append(copy.deepcopy(C))
#pdb.set_trace()
"1.3 add the system uncertainty"
A_t_env = []
B_t_env = []
C_t_env=  []
for time in range(batch_length):
    A_t_env.append(copy.deepcopy(A))
    #pdb.set_trace()
    A_t_env[time]=A_t[time]+A_t[time]*0.1*np.sin(time)
    #pdb.set_trace()
    B_t_env.append(copy.deepcopy(B))
    B_t_env[time] =B_t[time]+B_t[time] *0.2* np.cos(time)
    C_t_env.append(copy.deepcopy(C))
    C_t_env[time] = C_t[time] + C_t[time] * 0.05 * np.sin(2*time)
'1.3 add a additional parameter for the t+1'
C_t_env.append(copy.deepcopy(C))
C_t_env[batch_length] = C_t[batch_length] + C_t[batch_length] * 0.05 * np.sin(2*batch_length)

#pdb.set_trace()
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

#pdb.set_trace()
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
"2. load the initial control law"
mb_ocs = MBOCS(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t,H_t=H_t,Q_t=Q_t, R_t=R_t)
#mb_ocs.load_K(name='initial_control_law_final')
mb_ocs.control_law()

"3. set the simulated env"
'3.1 the initial state'
x_k0 = np.array([[5.], [5.] ,[5.]])
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
    z3 = np.array([x[2]])
    # the control signal u_{k,t}
    u1 = np.array([u[0]])
    u2 = np.array([u[1]])
    # Get the current time t state space
    A_t_env=params.get('A_t')
    B_t_env = params.get('B_t')
    # Compute the discrete updates
    dz1 = A_t_env[0,0]* z1 + A_t_env[0,1]* z2+A_t_env[0,2]* z3 +B_t_env[0,0]*u1+B_t_env[0,1]*u2
    dz2 = A_t_env[1,0]* z1 + A_t_env[1,1]* z2+A_t_env[1,2]* z3 +B_t_env[1,0]*u1+B_t_env[1,1]*u2
    dz3 = A_t_env[2, 0] * z1 + A_t_env[2, 1] * z2 + A_t_env[2, 2] * z3 + B_t_env[2, 0] * u1 + B_t_env[2, 1] * u2
    # pdb.set_trace()
    return [dz1, dz2, dz3]


def output_update(t, x, u, params):
    # Parameter setup
    # pdb.set_trace()
    # Compute the discrete updates
    # the state x_{k,t}
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    z3 = np.array([x[2]])
    # Get the current time t state space
    C_t_env=params.get('C_t')
    y1 = C_t_env[0,0]* z1 + C_t_env[0,1]* z2+C_t_env[0,2]* z3
    y2 = C_t_env[1, 0] * z1 + C_t_env[1, 1] * z2 + C_t_env[1, 2] * z3
    return [y1,y2]

batch_sys = control.NonlinearIOSystem(
    state_update, output_update, inputs=('u1','u2'), outputs=('y1','y2'),
    states=('dz1', 'dz2', 'dz3'), dt=1, name='linear_MIMO')

#controlled_system = TwoDimSys(batch_length=batch_length, sample_time=sample_time, sys=batch_sys, x_k0=x_k0, x_t0=x_t0)
#controlled_system = Time_varying_injection_molding(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t, B_t=B_t, C_t=C_t,D_t=D_t)
controlled_system = Time_varying_batch_system(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t_env, B_t=B_t_env, C_t=C_t_env)

#pdb.set_trace()
"4. initial the q-learning control scheme"
Q_learning=Q_learning_OCS(batch_length=batch_length,D_t=D_t,H_t=H_t,n=n,m=m,q=q,Q_t=Q_t,R_t=R_t)
Q_learning.initial_buffer(data_length=batch_explore)

"5. Sample the data"
# x,u, y
# define the buffer data
buffer_x=np.zeros((n,batch_length+1))
buffer_u=np.zeros((m,batch_length))
buffer_y=np.zeros((q,batch_length+1))


for batch in range(batch_explore):
    "5.1 reset the system"
    # each batch randomly initial state
    x_k0=np.array([[5.* (np.random.rand() * 2 - 1)],[5.* (np.random.rand() * 2 - 1)],[5.* (np.random.rand() * 2 - 1)]])
    #x_k0 = np.array((5.* (np.random.rand() * 2 - 1), 5. * (np.random.rand() * 2 - 1), 5. * (np.random.rand() * 2 - 1)))
    #pdb.set_trace()
    buffer_x[:, 0] = x_k0[:,0]
    #pdb.set_trace()
    x_tem, y_out = controlled_system.reset_randomly(xk_0=x_k0)
    #pdb.set_trace()
    #state_u = []
    #y_out_list = []
    #y_out_list.append(y_out)
    #pdb.set_trace()
    buffer_y[:, 0:1]=y_out[:,0]
    #pdb.set_trace()

    for time in range(batch_length):
        state = np.block([[x_tem],
                          [y_sum[time + 1]]])
        control_signal = -mb_ocs.K[time] @ state
        #tem = np.array([[5.* (np.random.rand() * 2 - 1)],[ 5. * (np.random.rand() * 2 - 1)]])
        tem = np.array([[1. * (np.random.rand() * 2 - 1)], [1. * (np.random.rand() * 2 - 1)]])
        control_signal = control_signal + tem
        x_tem, y_out = controlled_system.step(control_signal)
        #state_u.append(copy.deepcopy(control_signal))
        #y_out_list.append(copy.deepcopy(y_out))
        #pdb.set_trace()
        # save the buffer data
        buffer_x[:, time + 1] = x_tem[:,0]
        buffer_u[:, time:time+1] = control_signal[:,0]
        buffer_y[:, time+1] = y_out[:,0]
        #pdb.set_trace()
    Q_learning.save_data(x=buffer_x,u=buffer_u,y=buffer_y,y_sum=y_sum)
    #pdb.set_trace()
#pdb.set_trace()
Q_learning.save_buffer(num=batch_explore)
#pdb.set_trace()
Q_learning.load_buffer(num=batch_explore)
#pdb.set_trace()
"6. compute the q-learning optimal control scheme"

Q_learning.initial_control_policy(pi=mb_ocs.K)
pdb.set_trace()
Q_learning.q_learning_iteration(batch_num=batch_explore)

pdb.set_trace()
a=2