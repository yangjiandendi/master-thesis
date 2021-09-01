# coding:utf-8

# G_1 is gradient position array, G_1_o is gradient direction array

# define the partial derivative of drift "sin(bx) + ay" to x
D_X = b * np.cos(b*G_1[:,0]).reshape(G_1.shape[0],1)

# define the partial derivative of drift "sin(bx) + ay" to y
D_Y = a * np.ones([G_1.shape[0],1])

# define the interface points contribution to drift
D_I = (np.sin(b*ref_layer_points[:,0]) + a*ref_layer_points[:,1] \
       - np.sin(b*rest_layer_points[:,0]) - a*rest_layer_points[:,1]).reshape(3,1)

# build the dirft matrix
D = np.vstack([D_X , D_Y , D_I])
D_T = D.T

# build zero matrix
zero_matrix = np.zeros([1,1])

# concatenate drift matrix and kriging matrix
K_D = np.concatenate([np.concatenate([K,D],axis = 1),np.concatenate([D_T,zero_matrix],axis = 1)],axis = 0)

# build right side matrix of cokriging system
bk = np.concatenate([G_1_o[:,0],G_1_o[:,1],np.zeros(K_D.shape[0]-G_1.shape[0]*2)],axis = 0)
bk = np.reshape(bk,newshape = [bk.shape[0],1])

# solve kriging weight
w = np.linalg.lstsq(K_D,bk)[0]
