import pdb
import numpy as np
import pandas as pd
import copy
from scipy.optimize import lsq_linear
import os
class Q_learning_OCS:
    """
    Model-free Optimal control
        :param batch_length: The fixed batch length each iteration
        :param n: dimensions of the state variable
        :param m: dimensions of the input variable
        :param Q: state cost matrix
        :param R: input cost matrix
        :param F_0: initial value matrix
        :param K_t: current control policy
        :param k: after k iterations updating the control policy
    """
    def __init__(self, batch_length: int,D_t,H_t,n,m,q,Q_t,R_t):
        self.batch_length=batch_length
        # the dimensions of the state space
        self.n=n # the dimensions of the state variable
        self.m=m   # the dimensions of the input variable
        self.q=q  # the dimensions of the output variable
        # the cost function
        self.Q_t = Q_t
        self.R_t = R_t
        # the reference trajectory model
        self.D_t=D_t
        self.H_t=H_t

        # the initial control policy
        self.pi=[]
        for time in range(self.batch_length):
            self.pi.append(np.zeros([self.m,self.n+1]))
        self.pi_improved=copy.deepcopy(self.pi)


        # the Vaule-function matrix P_t^{i}
        self.P_t=[]
        for time in range(self.batch_length):
            self.P_t.append(np.zeros([self.n+1,self.n+1]))

        # the Q-function matrix M_t^{i}
        self.M_t=[]
        for time in range(self.batch_length):
            self.M_t.append(np.zeros([self.n+1+self.m,self.n+1+self.m]))


    def initial_buffer(self,data_length:int):

        # data structure:dimension: time length: batch length
        self.x = np.zeros((self.n, self.batch_length + 1, data_length))
        self.u = np.zeros((self.m,self.batch_length,data_length))
        self.y = np.zeros((self.q, self.batch_length+1, data_length))
        # the reference trajectory is identical
        self.y_sum = np.zeros((1, self.batch_length+1,data_length))
        self.data_index=0

    def initial_control_policy(self,pi):

        for time in range(self.batch_length):
            self.pi[time]=-pi[time]

    def save_data(self,x,u,y,y_sum):
        self.x[:,:,self.data_index]=x
        self.u[:,:,self.data_index]=u
        self.y[:, :, self.data_index] = y
        self.y_sum[:, :, self.data_index] = y_sum.T
        self.data_index+=1

    def save_buffer(self,path, num):
        # tranformate the 3D to 2D for save
        self.x = self.x.reshape(self.x.shape[0] * self.x.shape[1], self.x.shape[2])
        df_x = pd.DataFrame(self.x)
        #df_x.to_csv(path)
        df_x.to_csv('./Data/{}/Buffer_x_{}.csv'.format(path,num))
        self.u = self.u.reshape(self.u.shape[0] * self.u.shape[1], self.u.shape[2])
        df_u = pd.DataFrame(self.u)
        df_u.to_csv('./Data/{}/Buffer_u_{}.csv'.format(path,num))

        self.y = self.y.reshape(self.y.shape[0] * self.y.shape[1], self.y.shape[2])
        df_y = pd.DataFrame(self.y)
        df_y.to_csv('./Data/{}/Buffer_y_{}.csv'.format(path,num))

        self.y_sum = self.y_sum.reshape(self.y_sum.shape[0] * self.y_sum.shape[1], self.y_sum.shape[2])
        df_y_sum = pd.DataFrame(self.y_sum)
        df_y_sum.to_csv('./Data/{}/Buffer_y_sum_{}.csv'.format(path,num))
    def load_buffer(self,path, num):
        df_x = pd.read_csv('./Data/{}/Buffer_x_{}.csv'.format(path,num), index_col=0)
        self.x = df_x.to_numpy()
        self.x = self.x.reshape(self.n,int(self.x.shape[0]/self.n), num)

        df_u = pd.read_csv('./Data/{}/Buffer_u_{}.csv'.format(path,num), index_col=0)
        self.u = df_u.to_numpy()
        self.u = self.u.reshape(self.m, int(self.u.shape[0] / self.m), num)

        df_y = pd.read_csv('./Data/{}/Buffer_y_{}.csv'.format(path,num), index_col=0)
        self.y = df_y.to_numpy()
        self.y = self.y.reshape(self.m, int(self.y.shape[0] / self.m), num)

        df_y_sum = pd.read_csv('./Data/{}/Buffer_y_sum_{}.csv'.format(path,num), index_col=0)
        self.y_sum = df_y_sum.to_numpy()
        self.y_sum = self.y_sum.reshape(1, int(self.y_sum.shape[0] / 1), num)


    def q_learning_iteration(self,path,batch_num):
        '1. construct the x_bar,x_tilde, u'
        # x_bar= list[np.array[self.x+self.y_sum][batch_num],...,np.array[][batch_num]]
        # t= 0,1,2,3,...T-1
        x_bar = []
        # x_tilde=list[np.array[self.x+self.y+self.y_sum][batch_num],...,np.array[][batch_num]]
        # t= 1,2,3,...T
        x_tilde = []

        # u=list[np.array[m][batch_num],...,np.array[m][batch_num]]
        # t= 0,1,2,3,...T-1
        u = []
        for time in range(self.batch_length):
            x_bar.append(np.zeros([self.n+1, batch_num]))
            x_tilde.append(np.zeros([self.n+self.q+1, batch_num]))
            u.append(np.zeros([self.m, batch_num]))

        #pdb.set_trace()
        '2. assignment the x_bar,x_tilde, u'
        for time in range(self.batch_length):
            for batch in range(batch_num):
                # x_bar=[x_{k,t}, y_sum{t+1}]
                x_bar[time][:, batch] = np.block([self.x[:, time, batch],self.y_sum[:, time+1, batch]])
                # x_tilde=[x_{k,t+1}, y_{k,t+1}, y_sum{t+1}]
                x_tilde[time][:, batch] = np.block([self.x[:, time+1, batch], self.y[:, time+1, batch],self.y_sum[:, time+1, batch]])
                # u=[u_{k,t}]
                u[time][:, batch] = self.u[:, time, batch]
        '3. construct the psi*L=b for lsq_linear'
        psi_t = []
        b_t = []
        psi_dim_t = int(0.5 * (x_bar[time].shape[0] + self.m) * (x_bar[time].shape[0] + self.m + 1))
        for time in range(self.batch_length):
            psi_t.append(np.zeros([batch_num, psi_dim_t]))
            b_t.append(np.zeros(batch_num))
        # the dimensional of M_11 M_12 M_22
        M_11_dim_t = x_bar[0].shape[0]
        M_12_dim_1_t = x_bar[0].shape[0]
        M_12_dim_2_t = u[0].shape[0]
        M_22_dim_t =  u[0].shape[0]
        # the dimensional of vec H_11 H_12 H_22
        M_11_vec_dim_t = int(0.5 * M_11_dim_t * (M_11_dim_t + 1))
        M_12_vec_dim_t = M_12_dim_1_t * M_12_dim_2_t
        M_22_vec_dim_t = int(0.5 * M_22_dim_t * (M_22_dim_t + 1))

        '3.1 psi'
        # the psi is invariant
        for time in range(self.batch_length):
            for batch in range(batch_num):
                psi1 = self.symkronecker_product(x_bar[time][:, batch])
                psi2 = 2 * self.non_symkronecker_product(x_bar[time][:, batch], u[time][:, batch])
                psi3 = self.symkronecker_product(u[time][:, batch])
                psi = np.vstack((psi1, psi2, psi3))
                psi_t[time][batch] = psi[:, 0]
        '3.2 b'

        iteration_num = self.batch_length + 1 + 10

        for iteration in range(iteration_num):

            for time in range(self.batch_length - 1, -1, -1):

                if time == (self.batch_length - 1):
                    # The terminal matrix P_{T}==0
                    S_t = np.block([[np.zeros((self.n, self.n)), np.zeros((self.n, self.q)), np.zeros((self.n, 1))],
                                    [np.zeros((self.q, self.n)), self.Q_t[time], -self.Q_t[time] @ self.H_t[time + 1]],
                                    [np.zeros((1, self.n)), -self.H_t[time + 1].T @ self.Q_t[time],
                                     self.H_t[time + 1].T @ self.Q_t[time] @ self.H_t[time + 1]]])

                else:
                    S_t = np.block([[self.P_t[time+1][0:self.n, 0:self.n], np.zeros((self.n, self.q)),
                                     self.P_t[time+1][0:self.n, self.n:] @ self.D_t[time + 1]],
                                    [np.zeros((self.q, self.n)), self.Q_t[time], -self.Q_t[time] @ self.H_t[time + 1]],
                                    [self.D_t[time + 1].T @ self.P_t[time+1][self.n:, 0:self.n],
                                     -self.H_t[time + 1].T @ self.Q_t[time],
                                     self.H_t[time + 1].T @ self.Q_t[time] @ self.H_t[time + 1] + self.D_t[
                                         time + 1].T @ self.P_t[time+1][self.n:, self.n:] @ self.D_t[time + 1]]])

                "b_t=x_tilde@S_t@x_tilde+u_{k,t}R_tu_{k,t}"
                for batch in range(batch_num):
                    b_tem = x_tilde[time][:, batch].T @ S_t @ x_tilde[time][:, batch] + u[time][:,batch].T @ self.R_t[time] @ u[time][:,batch]
                    b_t[time][batch] = b_tem[0, 0]
                #rank = np.linalg.matrix_rank(psi_t[time])
                #print(rank)
                res = lsq_linear(psi_t[time], b_t[time], lsmr_tol='auto', verbose=1)


                "4. translate the vector into the matrix"
                M_vec_11 = res.x[0:M_11_vec_dim_t]
                M_vec_12 = res.x[M_11_vec_dim_t:(M_11_vec_dim_t + M_12_vec_dim_t)]
                M_vec_22 = res.x[(M_11_vec_dim_t + M_12_vec_dim_t):]



                M_11 = self.vector_to_symmatrix(M_vec_11, M_11_dim_t)
                M_12 = self.vector_to_non_symmatrix(M_vec_12, M_12_dim_1_t, M_12_dim_2_t)
                M_22 = self.vector_to_symmatrix(M_vec_22, M_22_dim_t)


                "4.1 construct the H_t^{i}"
                M_t = np.block([[M_11, M_12], [M_12.T, M_22]])

                self.M_t[time][:, :] = M_t[:, :]
                "5. policy improvement"
                K = np.linalg.inv(M_22) @ (M_12.T)
                "6. save the control law in the control policy"
                self.pi_improved[time] = copy.deepcopy(-K)
                "7. construct the cost function under the current control policy"
                tem = np.block([[np.eye(self.n + 1)], [self.pi[time]]])
                self.P_t[time] = tem.T @ M_t @ tem
            self.pi = copy.deepcopy(self.pi_improved)
            self.save_K_each_iteration(path=path,iteration_num=iteration+1, batch_num=batch_num)

    def save_K(self,batch_num):

        tem=self.pi_improved[0]
        for time in range(1,len(self.pi_improved)):
            tem=np.block([[tem,self.pi_improved[time]]])
        df = pd.DataFrame(tem)
        df.to_csv('./Data/control_law_off_policy_{}.csv'.format(batch_num))


    def load_K(self,path):

        config_path = os.path.split(os.path.abspath(__file__))[0]
        config_path = config_path.rsplit('/', 1)[0]
        dir_path = os.path.join(config_path, path)
        df_K = pd.read_csv(dir_path, index_col=0)
        tem_K=df_K.to_numpy()
        K_from_csv=[]
        tem_1=0
        tem_2=0
        for time in range(self.batch_length):
            tem_1=copy.deepcopy(tem_2)
            tem_2=tem_2+self.n+1
            K_from_csv.append(np.matrix(tem_K[:,tem_1:tem_2]))
        self.K=K_from_csv


    def symkronecker_product(self,vector):
        dim=vector.shape[0]
        tem = np.kron(vector, vector)
        tem = tem.reshape((dim, dim))
        tril=np.tril(tem)  # the low triangle of matrix
        triu=np.triu(tem)  # the upper triangle of matrix
        tem2=tem-triu+tril
        kronecked_product=np.zeros((int(0.5*dim*(dim+1)),1))
        num=0
        for column in range(dim):

            for row in range(dim):
                if row>=column:
                    kronecked_product[num,0]=tem2[row,column]
                    num=num+1
        return kronecked_product
    def non_symkronecker_product(self,vector1,vector2):
        dim1=vector1.shape[0]
        dim2=vector2.shape[0]
        tem = np.kron(vector2, vector1)
        kronecked_product=tem.reshape((dim1*dim2,1))
        return kronecked_product

    def vector_to_symmatrix(self,vector,dim):
        tem=np.zeros((dim,dim))
        num=0
        for column in range(dim):
            for row in range(dim):
                if row>=column:
                    tem[row,column]=vector[num]
                    num=num+1
        tem2=tem+tem.T
        tem3=np.diagflat(np.diag(tem))
        A_t=tem2-tem3
        return A_t
    def vector_to_non_symmatrix(self,vector,dim1,dim2):
        tem=np.zeros((dim1,dim2))
        num=0
        for column in range(dim2):
            for row in range(dim1):
                tem[row,column]=vector[num]
                num=num+1
        return tem

    def save_K_each_iteration(self,path,iteration_num,batch_num):
        config_path = os.path.split(os.path.abspath(__file__))[0]
        config_path = config_path.rsplit('/', 1)[0]
        K_tem=-self.pi[0]
        for time in range(1,self.batch_length):
            K_tem=np.block([[K_tem,-self.pi[time]]])
        df = pd.DataFrame(K_tem)

        dir_path = os.path.join(config_path, "Data/{}/policy_iteration_{}".format(path,batch_num))
        #dir_path = os.path.join(config_path, "Data/Injection_Molding/policy_iteration_{}".format(batch_num))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        data_path = os.path.join(dir_path, "q_learning_control_policy{}.csv".format(iteration_num))
        df.to_csv(data_path)


