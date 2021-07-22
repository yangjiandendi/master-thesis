import numpy as np

## defining the dips position
G_1 = np.array([[0., 1., 1.]])

G_1_x = 0
G_1_y = 1
G_1_z = 0

G_1_tiled = np.tile(G_1,[3,1])

def squared_euclidean_distance(x_1,x_2):
    sqd = np.sqrt(np.reshape(np.sum(x_1**2,1),newshape =(x_1.shape[0],1))+\
    np.reshape(np.sum(x_2**2,1),newshape =(1,x_2.shape[0]))-\
    2*(x_1@x_2.T))
    return sqd

def cartesian_dist(x_1,x_2):
    return np.concatenate([
        np.tile(x_1[:,0] - np.reshape(x_2[:,0],[x_2[:,0].shape[0],1]),[1,3]),
        np.tile(x_1[:,1] - np.reshape(x_2[:,1],[x_2[:,1].shape[0],1]),[1,3]),
        np.tile(x_1[:,2] - np.reshape(x_2[:,2],[x_2[:,2].shape[0],1]),[1,3])],axis = 0) 

h_u = cartesian_dist(G_1,G_1)
h_v = h_u.T

a = np.concatenate([np.ones([G_1.shape[0],G_1.shape[0]]),np.zeros([G_1.shape[0],G_1.shape[0]]),np.zeros([G_1.shape[0],G_1.shape[0]])],axis = 1)
b = np.concatenate([np.zeros([G_1.shape[0],G_1.shape[0]]),np.ones([G_1.shape[0],G_1.shape[0]]),np.zeros([G_1.shape[0],G_1.shape[0]])],axis = 1)
c = np.concatenate([np.zeros([G_1.shape[0],G_1.shape[0]]),np.zeros([G_1.shape[0],G_1.shape[0]]),np.ones([G_1.shape[0],G_1.shape[0]])],axis = 1)
perpendicularity_matrix = np.concatenate([a,b,c],axis = 0)

dist_tiled = squared_euclidean_distance(G_1_tiled,G_1_tiled)

a_T = 5
c_o_T = a_T**2/14/3

def cov_gradients(dist_tiled):
    
    condition1 = 0
    a = (h_u*h_v)
    b = dist_tiled**2

    t1 =  np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    t2 = -c_o_T*((-14/a_T**2)+
                 105/4*dist_tiled/a_T**3 -
                 35/2 * dist_tiled**3 / a_T **5 +
                 21 /4 * dist_tiled**5/a_T**7)+\
         c_o_T * 7 * (9 * dist_tiled ** 5 -
                      20 * a_T ** 2 * dist_tiled ** 3 +
                      15 * a_T ** 4 * dist_tiled -
                      4 * a_T ** 5) / (2 * a_T ** 7)

    # when we do the covariance of Gx with Gx, Gy with Gy and so on, quation B9 in Gempy paper
    t3 = perpendicularity_matrix * \
         c_o_T * ((-14 / a_T ** 2) + 105 / 4 * dist_tiled / a_T ** 3 -
                   35 / 2 * dist_tiled ** 3 / a_T ** 5 +
                   21 / 4 * dist_tiled ** 5 / a_T ** 7)
    t4 = 1/3*np.eye(dist_tiled.shape[0])

    condition2 = t1 * t2 - t3 + t4

    C_G = np.where(dist_tiled==0, condition1, condition2) ## adding nugget effect
    return C_G

dist_tiled = dist_tiled + np.eye(dist_tiled.shape[0])

C_G = cov_gradients(dist_tiled)

layer1 = np.array([[0,0,0],[0,2,2],[0,4,0]])
layer2 = np.array([[0,0,2],[0,2,4],[0,4,2]])
number_of_layer = 2
number_of_points_per_surface = np.array([layer1.shape[0],layer2.shape[0]])

def set_rest_ref_matrix(number_of_points_per_surface):
    ref_layer_points = np.repeat(np.stack([layer1[-1],layer2[-1]],axis = 0),repeats=number_of_points_per_surface-1,axis = 0)
    rest_layer_points = np.concatenate([layer1[0:-1],layer2[0:-1]],axis = 0)
    return ref_layer_points,rest_layer_points

ref_layer_points,rest_layer_points = set_rest_ref_matrix(number_of_points_per_surface)

sed_rest_rest = squared_euclidean_distance(rest_layer_points,rest_layer_points)
sed_ref_rest = squared_euclidean_distance(ref_layer_points,rest_layer_points)
sed_rest_ref = squared_euclidean_distance(rest_layer_points,ref_layer_points)
sed_ref_ref = squared_euclidean_distance(ref_layer_points,ref_layer_points)

def cov_interface(ref_layer_points,rest_layer_points):
    sed_rest_rest = squared_euclidean_distance(rest_layer_points,rest_layer_points)
    sed_ref_rest = squared_euclidean_distance(ref_layer_points,rest_layer_points)
    sed_rest_ref = squared_euclidean_distance(rest_layer_points,ref_layer_points)
    sed_ref_ref = squared_euclidean_distance(ref_layer_points,ref_layer_points)
    
    C_I = c_o_T*((1 - 7 * (sed_rest_rest / a_T) ** 2 +\
                35 / 4 * (sed_rest_rest / a_T) ** 3 -\
                7 / 2 * (sed_rest_rest / a_T) ** 5 +\
                3 / 4 * (sed_rest_rest / a_T) ** 7) -\
                (1 - 7 * (sed_ref_rest / a_T) ** 2 +\
                35 / 4 * (sed_ref_rest / a_T) ** 3 -\
                7 / 2 * (sed_ref_rest / a_T) ** 5 +\
                3 / 4 * (sed_ref_rest / a_T) ** 7) -\
                (1 - 7 * (sed_rest_ref / a_T) ** 2 +\
                35 / 4 * (sed_rest_ref / a_T) ** 3 -\
                7 / 2 * (sed_rest_ref / a_T) ** 5 +\
                3 / 4 * (sed_rest_ref / a_T) ** 7) +\
                (1 - 7 * (sed_ref_ref / a_T) ** 2 +\
                35 / 4 * (sed_ref_ref / a_T) ** 3 -\
                7 / 2 * (sed_ref_ref / a_T) ** 5 +\
                3 / 4 * (sed_ref_ref / a_T) ** 7))
    
    return C_I

C_I = cov_interface(ref_layer_points,rest_layer_points)

sed_dips_rest = squared_euclidean_distance(G_1_tiled,rest_layer_points)
sed_dips_ref = squared_euclidean_distance(G_1_tiled,ref_layer_points)

def cartesian_dist_no_tile(x_1,x_2):
    return np.concatenate([
        np.transpose((x_1[:,0] - np.reshape(x_2[:,0],[x_2.shape[0],1]))),
        np.transpose((x_1[:,1] - np.reshape(x_2[:,1],[x_2.shape[0],1]))),
        np.transpose((x_1[:,2] - np.reshape(x_2[:,2],[x_2.shape[0],1])))],axis = 0) 

hu_rest = cartesian_dist_no_tile(G_1,rest_layer_points)
hu_ref = cartesian_dist_no_tile(G_1,ref_layer_points)

def cov_interface_gradients(hu_rest,hu_ref):
    C_GI = (hu_rest*(- c_o_T * ((-14 / a_T ** 2) + 105 / 4 * sed_dips_rest / a_T ** 3 -
                                35 / 2 * sed_dips_rest ** 3 / a_T ** 5 +
                                21 / 4 * sed_dips_rest ** 5 / a_T ** 7))-\
    hu_ref*(-c_o_T * ((-14 / a_T ** 2) + 105 / 4 * sed_dips_ref / a_T ** 3 -
                                35 / 2 * sed_dips_ref ** 3 / a_T ** 5 +
                                21 / 4 * sed_dips_ref ** 5 / a_T ** 7)))
    return C_GI

C_GI = cov_interface_gradients(hu_rest,hu_ref)
C_IG = C_GI.T

K = np.concatenate([np.concatenate([C_G,C_GI],axis = 1),
np.concatenate([C_IG,C_I],axis = 1)],axis = 0)

xx = np.arange(-.5,4.5,0.1)
yy = np.arange(-.5,4.5,0.1)
zz = np.arange(-.5,4.5,0.1)
XX,YY,ZZ = np.meshgrid(xx,yy,zz)
X = (np.reshape(XX,[-1])).T
Y = (np.reshape(YY,[-1])).T
Z = (np.reshape(ZZ,[-1])).T
grid = np.stack([X,Y,Z],axis = 1)

hu_Simpoints = cartesian_dist_no_tile(G_1,grid)
sed_dips_SimPoint = squared_euclidean_distance(G_1_tiled,grid)

sed_rest_SimPoint = squared_euclidean_distance(rest_layer_points,grid)
sed_ref_SimPoint = squared_euclidean_distance(ref_layer_points,grid)

a1 = np.eye(3)

a2 = np.stack([ref_layer_points[:,0] - rest_layer_points[:,0],\
    ref_layer_points[:,1] - rest_layer_points[:,1],\
        ref_layer_points[:,2] - rest_layer_points[:,2]])

a3 = np.stack([[G_1[0,0]*2,0,0],[0,G_1[0,1]*2,0],[0,0,G_1[0,2]*2],\
            [G_1[0,1],G_1[0,0],0],[G_1[0,2],0,G_1[0,0]],[0,G_1[0,2],G_1[0,1]]])

a4 = np.stack([ref_layer_points[:,0]*ref_layer_points[:,0].T - rest_layer_points[:,0]*rest_layer_points[:,0].T , \
            ref_layer_points[:,1]*ref_layer_points[:,1].T - rest_layer_points[:,1]*rest_layer_points[:,1].T , \
            ref_layer_points[:,2]*ref_layer_points[:,2].T - rest_layer_points[:,2]*rest_layer_points[:,2].T, \
            ref_layer_points[:,0]*ref_layer_points[:,1] - rest_layer_points[:,0]*rest_layer_points[:,1],\
            ref_layer_points[:,0]*ref_layer_points[:,2] - rest_layer_points[:,0]*rest_layer_points[:,2],\
            ref_layer_points[:,1]*ref_layer_points[:,2] - rest_layer_points[:,1]*rest_layer_points[:,2]])

U_2nd = np.concatenate([np.concatenate([a1,a2],axis=1),np.concatenate([a3,a4],axis=1)], axis=0)

U_2nd_T = U_2nd.T

zero_matrix = np.zeros([9,9])

K_U_2nd = np.concatenate([np.concatenate([K,U_2nd_T],axis = 1),np.concatenate([U_2nd,zero_matrix],axis = 1)],axis = 0)

b_2nd = np.concatenate([[G_1_x,G_1_y,G_1_z],np.zeros(K_U_2nd.shape[0]-G_1.shape[0]*3)],axis = 0)

b_2nd = np.reshape(b_2nd,newshape = [b_2nd.shape[0],1])

w_2nd = np.linalg.lstsq(K_U_2nd,b_2nd)[0]

# gradient contribution
sigma_0_grad = w_2nd[:G_1.shape[0]*3] * (-hu_Simpoints*(- c_o_T * ((-14 / a_T ** 2) + 105 / 4 * sed_dips_SimPoint / a_T ** 3                -35 / 2 * sed_dips_SimPoint ** 3 / a_T ** 5 + 21 / 4 * sed_dips_SimPoint ** 5 / a_T ** 7)))

sigma_0_grad = np.sum(sigma_0_grad,axis=0)
# surface point contribution
sigma_0_interf = -w_2nd[G_1.shape[0]*3:-9]*(c_o_T  * ((1 - 7 * (sed_rest_SimPoint / a_T) ** 2 +
            35 / 4 * (sed_rest_SimPoint / a_T) ** 3 -
            7 / 2 * (sed_rest_SimPoint / a_T) ** 5 +
            3 / 4 * (sed_rest_SimPoint / a_T) ** 7) -
            (1 - 7 * (sed_ref_SimPoint / a_T) ** 2 +
            35 / 4 * (sed_ref_SimPoint / a_T) ** 3 -
            7 / 2 * (sed_ref_SimPoint / a_T) ** 5 +
            3 / 4 * (sed_ref_SimPoint / a_T) ** 7)))
sigma_0_interf = np.sum(sigma_0_interf,axis = 0)

# 2nd order drift contribution
sigma_0_2nd_drift_1 = np.sum(grid* (w_2nd[-9:-6]).T,axis = 1)

sigma_0_2nd_drift_xx = grid[:,0]*grid[:,0]* (w_2nd[-6]).T
sigma_0_2nd_drift_yy = grid[:,1]*grid[:,1]* (w_2nd[-5]).T
sigma_0_2nd_drift_zz = grid[:,2]*grid[:,2]* (w_2nd[-4]).T

sigma_0_2nd_drift_xy = grid[:,0]*grid[:,1]* (w_2nd[-3]).T
sigma_0_2nd_drift_xz = grid[:,0]*grid[:,2]* (w_2nd[-2]).T
sigma_0_2nd_drift_yz = grid[:,1]*grid[:,2]* (w_2nd[-1]).T

sigma_0_2nd_drift = sigma_0_2nd_drift_1 + sigma_0_2nd_drift_xx + sigma_0_2nd_drift_yy + sigma_0_2nd_drift_zz + sigma_0_2nd_drift_xy + sigma_0_2nd_drift_xz + sigma_0_2nd_drift_yz

interpolate_result = sigma_0_grad+sigma_0_interf+sigma_0_2nd_drift
intp = np.reshape(interpolate_result,[50,50,50]) 

print(intp)

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter(XX, YY, ZZ, c=intp)







