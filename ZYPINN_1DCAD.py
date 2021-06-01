
# Load required libraries
#import tensorflow as tf
import numpy as np
#import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1234)
#tf.random.set_seed(1234)
tf.set_random_seed(1234)
#tf.compat.v1.set_random_seed(1234)

tf.compat.v1.disable_eager_execution()

# The main solver class


class OneDBioPINN:
    # Initialize the class
    def __init__(self, X_measurement_aorta1, X_measurement_carotid,
                       X_measurement_aorta3, X_measurement_aorta4,
                       T_measurement, T_initial, 
                       A_training_aorta1,  U_training_aorta1,
                       A_training_carotid, U_training_carotid,
                       A_training_aorta3,  U_training_aorta3,
                       A_training_aorta4,  U_training_aorta4, 
                       X_residual_aorta1, 
                       X_residual_carotid, 
                       X_residual_aorta3, 
                       X_residual_aorta4,
                       T_residual,layers,bif_points):

        # Reference vessel areas
        self.A_01 = 2.293820e-04
        self.A_02 = 2.623127e-05
        self.A_03 = 2.411245e-04
        
        # Blood density and characteristic velocity
        self.rho = 1060.                   
        self.U = 1e+1

        # Charactiristic Variables for non-dimensionalization
        self.L = np.sqrt(0.333*(self.A_01 + self.A_02 + self.A_03))
        self.T = self.L/self.U
        self.p0 = self.rho*self.U**2        

        # Non-dimensionalize
        self.A0 = self.L**2     
        
        X_measurement_aorta1 = X_measurement_aorta1/self.L
        X_measurement_carotid = X_measurement_carotid/self.L
        X_measurement_aorta3 = X_measurement_aorta3/self.L
        X_measurement_aorta4 = X_measurement_aorta4/self.L
        
        X_residual_aorta1 = X_residual_aorta1/self.L
        X_residual_carotid = X_residual_carotid/self.L
        X_residual_aorta3 = X_residual_aorta3/self.L
        X_residual_aorta4 = X_residual_aorta4/self.L
        
        T_measurement  = T_measurement/self.T
        T_residual  = T_residual/self.T
        T_initial  = T_initial/self.T
        
        # Normalize inputs
        self.Xmean1, self.Xstd1 = X_residual_aorta1.mean(0), X_residual_aorta1.std(0)
        self.Xmean2, self.Xstd2 = X_residual_carotid.mean(0), X_residual_carotid.std(0)
        self.Xmean3, self.Xstd3 = X_residual_aorta3.mean(0), X_residual_aorta3.std(0)
        self.Xmean4, self.Xstd4 = X_residual_aorta4.mean(0), X_residual_aorta4.std(0)

        self.Tmean, self.Tstd = T_residual.mean(0), T_residual.std(0)
        
        # Jacobians
        self.jac_x1 = 1.0/self.Xstd1
        self.jac_x2 = 1.0/self.Xstd2
        self.jac_x3 = 1.0/self.Xstd3
        self.jac_x4 = 1.0/self.Xstd4

        self.jac_t = 1.0/self.Tstd
        
        # Store normalized/non-dimensionalized Variables
        self.X_f1 = (X_residual_aorta1 - self.Xmean1)/self.Xstd1
        self.X_u1 = (X_measurement_aorta1 - self.Xmean1)/self.Xstd1
        
        self.X_f2 = (X_residual_carotid - self.Xmean2)/self.Xstd2
        self.X_u2 = (X_measurement_carotid - self.Xmean2)/self.Xstd2
        
        self.X_f3 = (X_residual_aorta3 - self.Xmean3)/self.Xstd3
        self.X_u3 = (X_measurement_aorta3 - self.Xmean3)/self.Xstd3

        self.X_f4 = (X_residual_aorta4 - self.Xmean4)/self.Xstd4
        self.X_u4 = (X_measurement_aorta4 - self.Xmean4)/self.Xstd4

        self.T_u = (T_measurement - self.Tmean)/self.Tstd
        self.T_f = (T_residual - self.Tmean)/self.Tstd
        self.T_i = (T_initial - self.Tmean)/self.Tstd
        
        self.A_u1 = A_training_aorta1 
        self.u_u1 = U_training_aorta1
        
        self.A_u2 = A_training_carotid
        self.u_u2 = U_training_carotid

        self.A_u3 = A_training_aorta3 
        self.u_u3 = U_training_aorta3

        self.A_u4 = A_training_aorta4 
        self.u_u4 = U_training_aorta4
        
        X1_fm = bif_points[0]/self.L
        X2_fm = bif_points[0]/self.L
        X3_fm1 = bif_points[0]/self.L
        X3_fm2 = bif_points[1]/self.L
           
        bif_p1 = (X1_fm - self.Xmean1)/self.Xstd1
        bif_p2 = (X2_fm - self.Xmean2)/self.Xstd2
        bif_p31 = (X3_fm1 - self.Xmean3)/self.Xstd3
        bif_p32 = (X3_fm2 - self.Xmean3)/self.Xstd3
        bif_p4 = (X3_fm2 - self.Xmean4)/self.Xstd4        
       
        X1max = bif_p1[0]
        X2min = bif_p2[0]
        X3min = bif_p31[0]
        X3max = bif_p32[0]
        X4min = bif_p4[0]
        
        # Store neural net layer dimensions
        self.layers = layers
        
        # Initialize network weights and biases        
        self.weights1, self.biases1 = self.initialize_NN(layers)
        self.weights2, self.biases2 = self.initialize_NN(layers)
        self.weights3, self.biases3 = self.initialize_NN(layers)
        self.weights4, self.biases4 = self.initialize_NN(layers)
                       
                
      
        
    # Define placeholders and computational graph
    
     # self._states = tf.Variable(tf.ones(shape=[None, self._num_states]), dtype=tf.float32)
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
        
        self.X1_fm = tf.constant([X1max], shape = [1024,1], dtype=tf.float32)
        self.X2_fm = tf.constant([X2min], shape = [1024,1], dtype=tf.float32)
        self.X3_fml = tf.constant([X3min], shape = [1024,1], dtype=tf.float32)
        self.X3_fmu = tf.constant([X3max], shape = [1024,1], dtype=tf.float32)
        self.X4_fm = tf.constant([X4min], shape = [1024,1], dtype=tf.float32)
        
        self.A_u_tf1 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.A_u1.shape[1]))
        self.u_u_tf1 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.u_u1.shape[1]))
        
        self.A_u_tf2 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.A_u2.shape[1]))
        self.u_u_tf2 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.u_u2.shape[1]))

        self.A_u_tf3 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.A_u3.shape[1]))
        self.u_u_tf3 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.u_u3.shape[1]))

        self.A_u_tf4 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.A_u4.shape[1]))
        self.u_u_tf4 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.u_u4.shape[1]))
                
        self.X_u_tf1 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.X_u1.shape[1]))
        self.X_u_tf2 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.X_u2.shape[1]))
        self.X_u_tf3 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.X_u3.shape[1]))
        self.X_u_tf4 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.X_u4.shape[1]))
        
        self.t_u_tf = tf.compat.v1.placeholder(tf.float32,  shape=(None, self.T_u.shape[1]))
        self.t_i_tf = tf.compat.v1.placeholder(tf.float32,  shape=(None, self.T_i.shape[1]))

        self.X_f_tf1 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.X_f1.shape[1]))
        self.X_f_tf2 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.X_f2.shape[1]))
        self.X_f_tf3 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.X_f3.shape[1]))
        self.X_f_tf4 = tf.compat.v1.placeholder(tf.float32, shape=(None, self.X_f4.shape[1]))

        self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, self.T_f.shape[1]))
      
        # Neural net predictions
        self.A_u_pred1, self.u_u_pred1, _  = self.neural_net_aorta1(self.X_u_tf1, self.t_u_tf)
        self.A_u_pred2, self.u_u_pred2, _  = self.neural_net_carotid(self.X_u_tf2, self.t_u_tf)
        self.A_u_pred3, self.u_u_pred3, _  = self.neural_net_aorta3(self.X_u_tf3, self.t_i_tf)
        self.A_u_pred4, self.u_u_pred4, _  = self.neural_net_aorta4(self.X_u_tf4, self.t_u_tf)
        
        self.A_f_pred1, self.u_f_pred1, self.p_f_pred1  = self.neural_net_aorta1(self.X_f_tf1, self.t_f_tf)
        self.A_f_pred2, self.u_f_pred2, self.p_f_pred2  = self.neural_net_carotid(self.X_f_tf2, self.t_f_tf)
        self.A_f_pred3, self.u_f_pred3, self.p_f_pred3  = self.neural_net_aorta3(self.X_f_tf3, self.t_f_tf)
        self.A_f_pred4, self.u_f_pred4, self.p_f_pred4  = self.neural_net_aorta4(self.X_f_tf4, self.t_f_tf)
        
        # Compute PDE residuals
        self.r_A1, self.r_u1, self.r_p1  = self.pinn_aorta1(self.X_f_tf1, self.t_f_tf)
        self.r_A2, self.r_u2, self.r_p2  = self.pinn_carotid(self.X_f_tf2, self.t_f_tf)
        self.r_A3, self.r_u3, self.r_p3  = self.pinn_aorta3(self.X_f_tf3, self.t_f_tf)
        self.r_A4, self.r_u4, self.r_p4  = self.pinn_aorta4(self.X_f_tf4, self.t_f_tf)
            
        # Compute loss functions
        self.loss_A1, self.loss_u1                 = self.compute_measurement_loss_aorta1(self.A_u_pred1, self.u_u_pred1)
        self.loss_rA1, self.loss_ru1, self.loss_rp1 = self.compute_residual_loss_aorta1 (self.r_A1, self.r_u1, self.r_p1)
        
        self.loss_A2, self.loss_u2                 = self.compute_measurement_loss_carotid(self.A_u_pred2, self.u_u_pred2)
        self.loss_rA2, self.loss_ru2, self.loss_rp2 = self.compute_residual_loss_carotid (self.r_A2, self.r_u2, self.r_p2)

        self.loss_A3, self.loss_u3                 = self.compute_measurement_loss_aorta3(self.A_u_pred3, self.u_u_pred3)
        self.loss_rA3, self.loss_ru3, self.loss_rp3 = self.compute_residual_loss_aorta3 (self.r_A3, self.r_u3, self.r_p3)

        self.loss_A4, self.loss_u4                 = self.compute_measurement_loss_aorta4(self.A_u_pred4, self.u_u_pred4)
        self.loss_rA4, self.loss_ru4, self.loss_rp4 = self.compute_residual_loss_aorta4 (self.r_A4, self.r_u4, self.r_p4)
     
        self.loss_interface  = self.compute_interface_loss()
        
        self.loss_A = self.loss_A1 + self.loss_A2 + self.loss_A3 + self.loss_A4
        self.loss_u = self.loss_u1 + self.loss_u2 + self.loss_u3 + self.loss_u4
        
        self.loss_measurements = self.loss_A + self.loss_u
        
        self.loss_ru = self.loss_ru1 + self.loss_ru2 + self.loss_ru3 + self.loss_ru4
        self.loss_rA = self.loss_rA1 + self.loss_rA2 + self.loss_rA3 + self.loss_rA4
        self.loss_rp = self.loss_rp1 + self.loss_rp2 + self.loss_rp3 + self.loss_rp4
        self.loss_residual = self.loss_rA + self.loss_ru + self.loss_rp
        
        # Total loss
        self.loss = self.loss_interface + self.loss_residual  + self.loss_measurements
        
        # Define optimizer        
        self.optimizer  = tf.train.AdamOptimizer(self.learning_rate)

        self.train_op = self.optimizer.minimize(self.loss)
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = False
        # Define Tensorflow session
        self.sess = tf.Session(config=config)
        
        # Initialize Tensorflow Variables
        self.merged = tf.summary.merge_all()
        
        # Logger
        self.summary_writer = tf.summary.FileWriter('./logs', self.sess.graph)
        self.saver = tf.train.Saver()
        self.loss_value_log = [] 
        self.loss_A_log  = []
        self.loss_u_log = []
        self.loss_r_log = []
        self.loss_rp_log = []
        self.loss_c_log = []
        
       # init = tf.global_Variables_initializer()
        init = tf.initialize_all_variables()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
       
    # Neural net forward pass       
    def neural_net(self, H, weights, biases, layers):
        num_layers = len(layers)  
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_aorta1(self, x, t):
        Au = self.neural_net(tf.concat([x,t],1),self.weights1,self.biases1,self.layers)
        A = Au[:,0:1]
        u = Au[:,1:2]
        p = Au[:,2:3]
        return tf.exp(A), u, p
    
    def neural_net_carotid(self, x, t):
        Au = self.neural_net(tf.concat([x,t],1),self.weights2,self.biases2,self.layers)
        A = Au[:,0:1]
        u = Au[:,1:2]
        p = Au[:,2:3]
        return tf.exp(A), u, p
    
    def neural_net_aorta3(self, x, t):
        Au = self.neural_net(tf.concat([x,t],1),self.weights3,self.biases3,self.layers)
        A = Au[:,0:1]
        u = Au[:,1:2]
        p = Au[:,2:3]
        return tf.exp(A), u, p

    def neural_net_aorta4(self, x, t):
        Au = self.neural_net(tf.concat([x,t],1),self.weights4,self.biases4,self.layers)
        A = Au[:,0:1]
        u = Au[:,1:2]
        p = Au[:,2:3]
        return tf.exp(A), u, p
   
    def compute_interface_loss(self):
        
         A1, u1, p1 = self.neural_net_aorta1(self.X1_fm,self.t_f_tf) # A*, u*, p*
         
         A2, u2, p2 = self.neural_net_carotid(self.X2_fm,self.t_f_tf) # A*, u*, p*
         
         A3, u3, p3 = self.neural_net_aorta3(self.X3_fml,self.t_f_tf) # A*, u*, p*
         
         A3u, u3u, p3u = self.neural_net_aorta3(self.X3_fmu,self.t_f_tf) # A*, u*, p*

         A4, u4, p4 = self.neural_net_aorta4(self.X4_fm,self.t_f_tf) # A*, u*, p*
         
         Q1 = A1*u1
         Q2 = A2*u2
         Q3 = A3*u3
         
         loss_mass = tf.reduce_mean(tf.square((Q1 - Q2 - Q3))) 
         
         p_1 = p1 + (0.5*u1**2)
         p_2 = p2 + (0.5*u2**2)
         p_3 = p3 + (0.5*u3**2)
         
         loss_press = tf.reduce_mean(tf.square( p_1 - p_2)) + tf.reduce_mean(tf.square( p_1 - p_3))
                                
                         
         loss_C = tf.reduce_mean(tf.square((u3u - u4))) + \
                             tf.reduce_mean(tf.square((A3u - A4))) + tf.reduce_mean(tf.square( p3u - p4))
                             
         return  loss_mass + loss_press + loss_C
     
    # Linear tapering for the reference cross-sectional area
    def get_equilibrium_cross_sectional_area_aorta_1(self, x):
        x = self.L*(self.Xstd1*x + self.Xmean1)
        X1 = 0.
        X2 = 0.04964
        denom = X2-X1
        x1 = 2.293820e-04
        x2 = 2.636589e-04
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y

    def get_equilibrium_cross_sectional_area_carotid(self, x):
        x = self.L*(self.Xstd2*x + self.Xmean2)
        X1 = 0.04964
        X2 = 0.10284
        denom = X2-X1
        x1 = 2.636589e-04
        x2 = 2.623127e-05
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y

    def get_equilibrium_cross_sectional_area_aorta_3(self, x):
        x = self.L*(self.Xstd3*x + self.Xmean3)
        X1 = 0.04964
        X2 = 0.1383
        denom = X2-X1
        x1 = 2.636589e-04
        x2 = 2.177177e-04
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y

    def get_equilibrium_cross_sectional_area_aorta_4(self, x):
        x = self.L*(self.Xstd4*x + self.Xmean4)
        X1 = 0.1383
        X2 = 0.17056
        denom = X2-X1
        x1 = 2.177177e-04
        x2 = 2.411245e-04
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y

    def get_beta_aorta_1(self, x):
        x = self.L*(self.Xstd1*x + self.Xmean1)
        X1 = 0.
        X2 = 0.04964
        denom = X2-X1
        x1 = 2.472667e+06
        x2 = 2.151208e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
    
    def get_beta_carotid(self, x):
        x = self.L*(self.Xstd2*x + self.Xmean2)
        X1 = 0.04964
        X2 = 0.10284
        denom = X2-X1
        x1 =2.151208e+06
        x2 = 9.459836e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
    
    def get_beta_aorta_3(self, x):
        x = self.L*(self.Xstd3*x + self.Xmean3)
        X1 = 0.04964
        X2 = 0.1383
        denom = X2-X1
        x1 = 2.151208e+06
        x2 = 2.800526e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
    
    def get_beta_aorta_4(self, x):
        x = self.L*(self.Xstd4*x + self.Xmean4)
        X1 = 0.1383
        X2 = 0.17056
        denom = X2-X1
        x1 = 2.800526e+06
        x2 = 2.528670e+06
        numer =  x2 - x1 
        alpha = numer/denom
        beta = x1 - alpha*X1
        y = alpha*x + beta
        return y
     
    # Compute residuals
    def pinn_aorta1(self, x, t):
        
        A, u, p = self.neural_net_aorta1(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_01 = self.get_equilibrium_cross_sectional_area_aorta_1(x)
        beta1 = self.get_beta_aorta_1(x)
        
        r_p  = 10000. + beta1*(tf.sqrt(A*self.A0) - tf.sqrt(A_01)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x1

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x1
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x1
                
        r_A = A_t + u*A_x + A*u_x 
        r_u = u_t + p_x + u*u_x 
        
        return r_A, r_u, r_p
    
    def pinn_carotid(self, x, t):
        
        A, u, p = self.neural_net_carotid(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_02 = self.get_equilibrium_cross_sectional_area_carotid(x)
        beta2 = self.get_beta_carotid(x)
        
        r_p  = 8.5e+3 + beta2*(tf.sqrt(A*self.A0) - tf.sqrt(A_02)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x2

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x2
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x2
                
        r_A = A_t + u*A_x +  A*u_x 
        r_u = u_t + p_x + u*u_x 
        
        return r_A, r_u, r_p
    
    def pinn_aorta3(self, x, t):
        
        A, u, p = self.neural_net_aorta3(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_03 = self.get_equilibrium_cross_sectional_area_aorta_3(x)
        beta3 = self.get_beta_aorta_3(x)

        r_p  = 10000. + beta3*(tf.sqrt(A*self.A0) - tf.sqrt(A_03)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x3

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x3
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x3
                
        r_A = A_t + u*A_x + A*u_x 
        r_u = u_t + p_x + u*u_x 
        
        return r_A, r_u, r_p

    def pinn_aorta4(self, x, t):
        
        A, u, p = self.neural_net_aorta4(x,t) # \hat{A}, \hat{u}, \hat{p}
        
        A_04 = self.get_equilibrium_cross_sectional_area_aorta_4(x)
        beta4 = self.get_beta_aorta_4(x)
        
        r_p  = 10000. + beta4*(tf.sqrt(A*self.A0) - tf.sqrt(A_04)) 
        
        p_x = tf.gradients(p, x)[0]*self.jac_x4

        A_t = tf.gradients(A, t)[0]*self.jac_t
        A_x = tf.gradients(A, x)[0]*self.jac_x4
        
        u_t = tf.gradients(u, t)[0]*self.jac_t
        u_x = tf.gradients(u, x)[0]*self.jac_x4
                
        r_A = A_t + u*A_x + A*u_x 
        r_u = u_t + p_x + u*u_x 
        
        return r_A, r_u, r_p

    # Compute residual losses
    def compute_residual_loss_aorta1(self, r_A, r_u, r_p):
        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred1 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp

    def compute_residual_loss_carotid(self, r_A, r_u, r_p):
        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred2 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp

    def compute_residual_loss_aorta3(self, r_A, r_u, r_p):
        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred3 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp 

    def compute_residual_loss_aorta4(self, r_A, r_u, r_p):
        loss_rA = tf.reduce_mean(tf.square(r_A)) 
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred4 - r_p*(1/self.p0))))

        return  loss_rA, loss_ru, loss_rp 

    def compute_measurement_loss_aorta1(self, A_u, u_u):
    
        loss_A = tf.reduce_mean(tf.square((self.A_u1 - A_u*self.A0)/self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u1 - u_u*self.U)/self.U))

        return loss_A, loss_u

    def compute_measurement_loss_carotid(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u2 - A_u*self.A0)/self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u2 - u_u*self.U)/self.U))

        return loss_A, loss_u

    def compute_measurement_loss_aorta3(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u3 - A_u*self.A0)/self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u3 - u_u*self.U)/self.U))

        return loss_A, loss_u

    def compute_measurement_loss_aorta4(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u4 - A_u*self.A0)/self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u4 - u_u*self.U)/self.U))

        return loss_A, loss_u
      
    # Fetch a mini-batch of data for stochastic gradient updates
    def fetch_minibatch(self, X1_f, X2_f, X3_f ,X4_f, t_f, N_f_batch):        
        N_f = X1_f.shape[0]
        idx_f = np.random.choice(N_f, N_f_batch, replace=False)
        X1_f_batch = X1_f[idx_f,:]
        X2_f_batch = X2_f[idx_f,:]
        X3_f_batch = X3_f[idx_f,:]
        X4_f_batch = X4_f[idx_f,:]

        t_f_batch = t_f[idx_f,:]        
        return  X1_f_batch, X2_f_batch, X3_f_batch, X4_f_batch, t_f_batch
             
    # Trains the model by minimizing the MSE loss using mini-batch stochastic gradient descent
    def train(self, nIter = 20000, batch_size = 1024, learning_rate = 1e-3): 

#         start_time = timeit.default_timer()
        for it in tqdm(range(nIter)):
            
            # Fetch a mini-batch of training data
            X1_f_batch, X2_f_batch, X3_f_batch, X4_f_batch, T_f_batch = self.fetch_minibatch(self.X_f1, 
                                                                                             self.X_f2, 
                                                                                             self.X_f3, 
                                                                                             self.X_f4, 
                                                                                             self.T_f,
                                                                                             N_f_batch = batch_size)
            # Define a dictionary for associating Variables with data
            tf_dict = {self.X_u_tf1: self.X_u1,  
                       self.X_u_tf2: self.X_u2, 
                       self.X_u_tf3: self.X_u3, 
                       self.X_u_tf4: self.X_u4, 
                       self.X_f_tf1: X1_f_batch,
                       self.X_f_tf2: X2_f_batch, 
                       self.X_f_tf3: X3_f_batch,
                       self.X_f_tf4: X4_f_batch,
                       self.t_f_tf:  T_f_batch, 
                       self.t_u_tf:  self.T_u,
                       self.t_i_tf:  self.T_i,
                       self.A_u_tf1: self.A_u1, self.u_u_tf1: self.u_u1, 
                       self.A_u_tf2: self.A_u2, self.u_u_tf2: self.u_u2,
                       self.A_u_tf3: self.A_u3, self.u_u_tf3: self.u_u3,
                       self.A_u_tf4: self.A_u4, self.u_u_tf4: self.u_u4,
                       self.learning_rate: learning_rate}
            
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)
            
            # Print
            if it % 50 == 0:
#                 elapsed = timeit.default_timer() - start_time
                loss_value, loss_A, loss_u, loss_r, loss_rp, loss_c  = self.sess.run([self.loss, 
                                                                                      self.loss_A,
                                                                                      self.loss_u, 
                                                                                      self.loss_ru+self.loss_rA, 
                                                                                      self.loss_rp, 
                                                                                      self.loss_interface], tf_dict)
                self.loss_value_log.append(loss_value) 
                self.loss_A_log.append(loss_A) 
                self.loss_u_log.append(loss_u) 
                self.loss_r_log.append(loss_r) 
                self.loss_rp_log.append(loss_rp) 
                self.loss_c_log.append(loss_c) 
#                 print('It: %d, Loss: %.3e, Loss_A: %.3e, Loss_u: %.3e, Loss_r: %.3e, Loss_p: %.3e\
#                                            Loss_c: %.3e, Time: %.2f' % 
#                       (it, loss_value, loss_A, loss_u, loss_r, loss_rp, loss_c, elapsed))
#                 start_time = timeit.default_timer()
                                
    # Evaluates predictions at test points           
    def predict_aorta1(self, X1, t): 
        # non-dimensionalize inputs
        X1 = X1/self.L
        t  = t/self.T
        # normalize inputs
        X1 = (X1 - self.Xmean1)/self.Xstd1
        t = (t - self.Tmean)/self.Tstd
        # Create tf dictionary
        tf_dict1 = {self.X_f_tf1: X1, self.t_f_tf: t}    
        # Evaluate predictions
        A_star1 = self.sess.run(self.A_f_pred1, tf_dict1) 
        u_star1 = self.sess.run(self.u_f_pred1, tf_dict1) 
        p_star1 = self.sess.run(self.p_f_pred1, tf_dict1) 
        # de-normalize outputs        
        A_star1 = A_star1*self.A0
        u_star1 = u_star1*self.U
        p_star1 = p_star1*self.p0
              
        return A_star1, u_star1, p_star1

    def predict_carotid(self, X2, t):     
        X2 = X2/self.L
        t  = t/self.T

        X2 = (X2 - self.Xmean2)/self.Xstd2

        t = (t - self.Tmean)/self.Tstd        
        tf_dict2 = {self.X_f_tf2: X2, self.t_f_tf: t}    
       
        A_star2 = self.sess.run(self.A_f_pred2, tf_dict2) 
        u_star2 = self.sess.run(self.u_f_pred2, tf_dict2) 
        p_star2 = self.sess.run(self.p_f_pred2, tf_dict2) 
                
        A_star2 = A_star2*self.A0
        u_star2 = u_star2*self.U
        p_star2 = p_star2*self.p0
              
        return A_star2, u_star2, p_star2
    
    def predict_aorta3(self, X3, t):     
        X3 = X3/self.L
        t  = t/self.T

        X3 = (X3 - self.Xmean3)/self.Xstd3
        t = (t - self.Tmean)/self.Tstd
        
        tf_dict3 = {self.X_f_tf3: X3, self.t_f_tf: t}    
       
        A_star3 = self.sess.run(self.A_f_pred3, tf_dict3) 
        u_star3 = self.sess.run(self.u_f_pred3, tf_dict3) 
        p_star3 = self.sess.run(self.p_f_pred3, tf_dict3) 
                
        A_star3 = A_star3*self.A0
        u_star3 = u_star3*self.U
        p_star3 = p_star3*self.p0
              
        return A_star3, u_star3, p_star3
    
    def predict_aorta4(self, X4, t):     
        X4 = X4/self.L
        t  = t/self.T

        X4 = (X4 - self.Xmean4)/self.Xstd4
        t = (t - self.Tmean)/self.Tstd
        
        tf_dict4 = {self.X_f_tf4: X4, self.t_f_tf: t}    
       
        A_star4 = self.sess.run(self.A_f_pred4, tf_dict4) 
        u_star4 = self.sess.run(self.u_f_pred4, tf_dict4) 
        p_star4 = self.sess.run(self.p_f_pred4, tf_dict4) 
                
        A_star4 = A_star4*self.A0
        u_star4 = u_star4*self.U
        p_star4 = p_star4*self.p0
              
        return A_star4, u_star4, p_star4

# Helper functions for setting up the problem
def get_equilibrium_cross_sectional_area_aorta_1(x):
    X1 = 0.0
    X2 = 0.04964
    denom = X2-X1
    x1 = 2.293820e-04
    x2 = 2.636589e-04
    numer =  x2 - x1 
    alpha = numer/denom
    beta = x1 - alpha*X1
    y = alpha*x + beta
    return y

def get_equilibrium_cross_sectional_area_carotid(x):
    X1 = 0.04964
    X2 = 0.10284
    denom = X2-X1
    x1 = 2.636589e-04
    x2 = 2.623127e-05
    numer =  x2 - x1 
    alpha = numer/denom
    beta = x1 - alpha*X1
    y = alpha*x + beta
    return y

def get_equilibrium_cross_sectional_area_aorta_3(x):
    X1 = 0.04964
    X2 = 0.1383
    denom = X2-X1
    x1 = 2.636589e-04
    x2 = 2.177177e-04
    numer =  x2 - x1 
    alpha = numer/denom
    beta = x1 - alpha*X1
    y = alpha*x + beta
    return y

def get_equilibrium_cross_sectional_area_aorta_4(x):
    X1 = 0.1383
    X2 = 0.17056
    denom = X2-X1
    x1 = 2.177177e-04
    x2 = 2.411245e-04
    numer =  x2 - x1 
    alpha = numer/denom
    beta = x1 - alpha*X1
    y = alpha*x + beta
    return y


# Define the number of spatio-temporal domain points to evaluate the residual
# of the system of equations.

N_f =  2000

aorta1_velocity = np.load("../data/Aorta1_U.npy", allow_pickle=True).item()
aorta2_velocity = np.load("../data/Aorta2_U.npy", allow_pickle=True).item()
aorta4_velocity = np.load("../data/Aorta4_U.npy", allow_pickle=True).item()
carotid_velocity= np.load("../data/LCommonCarotid_U.npy", allow_pickle=True).item()

aorta1_area = np.load("../data/Aorta1_A.npy", allow_pickle=True).item()
aorta2_area = np.load("../data/Aorta2_A.npy", allow_pickle=True).item()
aorta4_area = np.load("../data/Aorta4_A.npy", allow_pickle=True).item()
carotid_area = np.load("../data/LCommonCarotid_A.npy", allow_pickle=True).item()

test_aorta3_velocity = np.load("../data/Aorta3_U.npy", allow_pickle=True).item()
test_aorta3_area = np.load("../data/Aorta3_A.npy", allow_pickle=True).item()

t = aorta1_velocity['t']*1e-3

velocity_measurements_aorta1 = aorta1_velocity["U"]*1e-2
velocity_measurements_carotid = carotid_velocity["U"]*1e-2
velocity_measurements_aorta4 = aorta4_velocity["U"]*1e-2

area_measurements_aorta1 = aorta1_area["A"]*1e-6
area_measurements_carotid = carotid_area["A"]*1e-6
area_measurements_aorta4 = aorta4_area["A"]*1e-6

velocity_testpoint_aorta3 = test_aorta3_velocity["U"]*1e-2
area_testpoint_aorta3 = test_aorta3_area["A"]*1e-6

u_test1 = aorta2_velocity['U']*1e-2
A_test1 = aorta2_area['A']*1e-6

# Number of measurements

N_u = t.shape[0]

layers = [2, 100, 100, 100, 100, 100, 100, 3]

lower_bound_t = t.min(0)
upper_bound_t = t.max(0)

lower_bound_vessel_1 = 0.0   
upper_bound_vessel_1 = 0.04964

lower_bound_vessel_2 = 0.04964
upper_bound_vessel_2 = 0.10284

lower_bound_vessel_3 = 0.04964
upper_bound_vessel_3 = 0.1383

lower_bound_vessel_4 = 0.1383
upper_bound_vessel_4 = 0.17056

# Spatial/temporal coordinates for initial conditions
X_initial_aorta1 = np.linspace(lower_bound_vessel_1,upper_bound_vessel_1,N_u)[:,None]
X_initial_carotid = np.linspace(lower_bound_vessel_2,upper_bound_vessel_2,N_u)[:,None]
X_initial_aorta3 = np.linspace(lower_bound_vessel_3,upper_bound_vessel_3,N_u)[:,None]
X_initial_aorta4 = np.linspace(lower_bound_vessel_4,upper_bound_vessel_4,N_u)[:,None]

T_initial  = lower_bound_t*np.ones((N_u))[:,None]

# Spatial/temporal coordinates for boundary conditions
X_boundary_aorta1 = lower_bound_vessel_1*np.ones((N_u))[:,None]
X_boundary_carotid = upper_bound_vessel_2*np.ones((N_u))[:,None]
X_boundary_aorta3 = upper_bound_vessel_3*np.ones((N_u))[:,None]
X_boundary_aorta4 = upper_bound_vessel_4*np.ones((N_u))[:,None]

T_boundary = t

# Measurement Spatial/temporal coordinates
X_measurement_aorta1 = np.vstack((X_initial_aorta1, X_boundary_aorta1))
X_measurement_carotid = np.vstack((X_initial_carotid, X_boundary_carotid))    
X_measurement_aorta3 = np.vstack((X_initial_aorta3))    
X_measurement_aorta4 = np.vstack((X_initial_aorta4, X_boundary_aorta4))    

T_measurement = np.vstack((T_initial, T_boundary))

X_residual_aorta1 = lower_bound_vessel_1 + (upper_bound_vessel_1-lower_bound_vessel_1)*np.random.random((N_f))[:,None]
X_residual_carotid = lower_bound_vessel_2 + (upper_bound_vessel_2-lower_bound_vessel_2)*np.random.random((N_f))[:,None]
X_residual_aorta3 = lower_bound_vessel_3 + (upper_bound_vessel_3-lower_bound_vessel_3)*np.random.random((N_f))[:,None]
X_residual_aorta4 = lower_bound_vessel_4 + (upper_bound_vessel_4-lower_bound_vessel_4)*np.random.random((N_f))[:,None]

T_residual = lower_bound_t + (upper_bound_t-lower_bound_t)*np.random.random((N_f))[:,None]

A_initial_aorta1 = get_equilibrium_cross_sectional_area_aorta_1(X_initial_aorta1)
A_initial_carotid = get_equilibrium_cross_sectional_area_carotid(X_initial_carotid)
A_initial_aorta3 = get_equilibrium_cross_sectional_area_aorta_3(X_initial_aorta3)    
A_initial_aorta4 = get_equilibrium_cross_sectional_area_aorta_4(X_initial_aorta4)

U_initial_aorta1 = velocity_measurements_aorta1[0]*np.ones((N_u,1))
U_initial_aorta2 = velocity_measurements_carotid[0]*np.ones((N_u,1))
U_initial_aorta3 = velocity_testpoint_aorta3[0]*np.ones((N_u,1))
U_initial_aorta4 = velocity_measurements_aorta4[0]*np.ones((N_u,1))

A_training_aorta1 = np.vstack((A_initial_aorta1,area_measurements_aorta1))
U_training_aorta1 = np.vstack((U_initial_aorta1,velocity_measurements_aorta1))

A_training_carotid = np.vstack((A_initial_carotid,area_measurements_carotid))
U_training_carotid = np.vstack((U_initial_aorta2,velocity_measurements_carotid))

A_training_aorta3 = np.vstack((A_initial_aorta3))
U_training_aorta3 = np.vstack((U_initial_aorta3))

A_training_aorta4 = np.vstack((A_initial_aorta4,area_measurements_aorta4))
U_training_aorta4 = np.vstack((U_initial_aorta4,velocity_measurements_aorta4))

bif_points = [upper_bound_vessel_1, upper_bound_vessel_3]


# Build the PINN model
model = OneDBioPINN(X_measurement_aorta1, X_measurement_carotid,
                    X_measurement_aorta3, X_measurement_aorta4,
                    T_measurement, T_initial, 
                    A_training_aorta1,  U_training_aorta1,
                    A_training_carotid, U_training_carotid,
                    A_training_aorta3,  U_training_aorta3,
                    A_training_aorta4,  U_training_aorta4,
                    X_residual_aorta1,
                    X_residual_carotid,
                    X_residual_aorta3, 
                    X_residual_aorta4,
                    T_residual,
                    layers,
                    bif_points)


# Load pre-trained model
model.saver.restore(model.sess, '../SavedModels/CarotidBifurcation.ckpt')

# Train the PINN model using mini-batch stochastic gradient descent
model.train(nIter = 200, batch_size = 1024, learning_rate = 1e-4)

# Evaluate model predictions at the test locations
test_point1 = 0.04964*np.ones((X_residual_aorta1.shape[0],1))    
test_point3 = 0.1383*np.ones((t.shape[0],1))

test_aorta1_lboundary = lower_bound_vessel_1*np.ones((t.shape[0],1))
test_carotid_rboundary = upper_bound_vessel_2*np.ones((t.shape[0],1))
test_aorta4_rboundary = upper_bound_vessel_4*np.ones((t.shape[0],1))

A_predict_aorta1, u_predict_aorta1, p_predict_aorta1     = model.predict_aorta1(test_point1, T_residual)
A_predict_carotid, u_predict_carotid, p_predict_carotid    = model.predict_carotid(test_point1, T_residual)
A_predict_aorta3l, u_predict_aorta3l, p_predict_aorta3l  = model.predict_aorta3(test_point1, T_residual)
A_predict_aorta4, u_predict_aorta4, p_predict_aorta4  = model.predict_aorta4(test_point3, t)

A_pred1b, u_pred1b, p_pred1b  = model.predict_aorta1(test_aorta1_lboundary, t)
A_pred2b, u_pred2b, p_pred2b  = model.predict_carotid(test_carotid_rboundary, t)
A_pred3b, u_pred3b, p_pred3b  = model.predict_aorta4(test_aorta4_rboundary, t)

# Plot the results
fig1 = plt.figure(1,figsize=(10, 6), facecolor='w')
fig2 = plt.figure(2,figsize=(10, 6), facecolor='w')
fig3 = plt.figure(3,figsize=(10, 6), facecolor='w')

ax1 = fig1.add_subplot(111)  
ax2 = fig2.add_subplot(111)  
ax3 = fig3.add_subplot(111)  

ax1.plot(t, u_predict_aorta4,'r--',linewidth=3.5, markersize=2.5)
ax1.plot(t, velocity_testpoint_aorta3,'b-',linewidth=3.5, markersize=2.5)

ax1.set_xlabel('Time in $s$')
ax1.set_ylabel('Velocity in $m/s$')
ax1.set_title('Compare velocity aorta3')

ax2.plot(t, A_predict_aorta4,'r--',linewidth=3.5, markersize=2.5)
ax2.plot(t, area_testpoint_aorta3,'b-',linewidth=3.5, markersize=2.5)

ax2.set_xlabel('Time in $s$')
ax2.set_ylabel('Area in $mm^2$')
ax2.set_title('Compare area aorta3')

ax3.plot(t, p_predict_aorta4/133.,'r--',linewidth=3.5, markersize=2.5)

ax3.set_xlabel('Time in $s$')
ax3.set_ylabel('Pressure in $mmHg$')
ax3.set_title('Pressure aorta3')
