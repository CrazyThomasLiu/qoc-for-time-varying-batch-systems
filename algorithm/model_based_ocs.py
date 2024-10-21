import pdb
import typing
import numpy as np
import copy
import pandas as pd
import csv

# Model-based optimal control law
class MBOCS:

    def __init__(self, batch_length: int, A_t, B_t, C_t, D_t,H_t,Q_t, R_t):
        self.batch_length=batch_length
        # the state space
        self.A_t = A_t
        self.B_t = B_t
        self.C_t = C_t
        # the reference trajectory model
        self.D_t=D_t
        self.H_t=H_t
        # the dimensions of the state space
        self.n=self.A_t[0].shape[0] # the dimensions of the state variable
        self.m=self.B_t[0].shape[1]   # the dimensions of the input variable
        self.q=self.C_t[0].shape[0]  # the dimensions of the output variable
        # the weighting matrix
        self.Q_t = Q_t
        self.R_t = R_t
        # save the all K_t and P_t
        self.K=[]
        self.P=[]

    def control_law(self):
        for time in range(self.batch_length - 1, -1, -1):  # [T-1,T-2,...,1,0,-1)
            # E_t matrix
            E_t=np.block([[self.A_t[time],np.zeros((self.n, 1))],
                          [self.C_t[time+1] @ self.A_t[time],np.zeros((self.q, 1))],
                          [np.zeros((1, self.n)),np.eye(1)]])

            # F_t matrix
            F_t=np.block([[self.B_t[time]],
                            [self.C_t[time+1] @ self.B_t[time]],
                            [np.zeros((1, self.m))]])
            if time== (self.batch_length - 1):

                # S_t
                S_t = np.block([[np.zeros((self.n, self.n)), np.zeros((self.n, self.q)), np.zeros((self.n, 1))],
                                [np.zeros((self.q, self.n)), self.Q_t[time], -self.Q_t[time]@self.H_t[time+1]],
                                [np.zeros((1, self.n)), -self.H_t[time+1].T@self.Q_t[time], self.H_t[time+1].T@self.Q_t[time]@self.H_t[time+1]]])

            else:
                S_t=np.block([[P_t[0:self.n,0:self.n],np.zeros((self.n, self.q)),P_t[0:self.n,self.n:]@self.D_t[time+1]],
                              [np.zeros((self.q, self.n)),self.Q_t[time],-self.Q_t[time]@self.H_t[time+1]],
                              [self.D_t[time+1].T@P_t[self.n:,0:self.n],-self.H_t[time+1].T@self.Q_t[time],self.H_t[time+1].T@self.Q_t[time]@self.H_t[time+1]+self.D_t[time+1].T@P_t[self.n:,self.n:]@self.D_t[time+1]]])


            tem=self.R_t[time]+F_t.T@S_t@F_t
            tem=np.linalg.inv(tem)
            K_t=tem@F_t.T@S_t@E_t
            P_t=E_t.T@(S_t-S_t@F_t@tem@F_t.T@S_t)@E_t
            # save the optimal control policy and the optimal cost function
            self.K.append(copy.deepcopy(K_t))
            self.P.append(copy.deepcopy(P_t))


        self.K.reverse()
        self.P.reverse()



    def save_K(self,name):

        tem=self.K[0]
        #pdb.set_trace()
        for time in range(1,len(self.K)):
            tem=np.block([[tem,self.K[time]]])
        df = pd.DataFrame(tem)
        #pdb.set_trace()
        df.to_csv('./Data/{}.csv'.format(name))



    def load_K(self,name):
        df_K = pd.read_csv('./Data/{}.csv'.format(name), index_col=0)
        tem_K=df_K.to_numpy()
        K_from_csv=[]
        tem_1=0
        tem_2=0
        for time in range(self.batch_length):
            tem_1=copy.deepcopy(tem_2)
            tem_2=tem_2+self.n+1
            K_from_csv.append(np.matrix(tem_K[:,tem_1:tem_2]))
        self.K=K_from_csv
