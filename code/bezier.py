# coding:utf-8

# G_1 is gradient position array, G_1_o is gradient direction array

def get_coef(x1,x2,x3,y1,y2,y3):
    """
    Convert quadratic Bézier curve to quadratic polynomials

    Args:
        x1, x2, x3: x coodinates of control points
        y1, y2, y3: y coodinates of control points
        
    Returns:
        a, b, c, d, e: coefficient of quadratic polynomials
    """
    a = 2*x1*y1*y3 - 4*x1*y2**2 + 4*x1*y2*y3 -2*x1*y3**2 + 4*x2*y1*y2 - \
        8*x2*y1*y3 + 4*x2*y2*y3 - 2*x3*y1**2 + 4*x3*y1*y2 + 2*x3*y1*y3 - 4*x3*y2**2 
    b = -2*x1**2*y3 + 4*x1*x2*y2 + 4*x1*x2*y3 + 2*x1*x3*y1 - 8*x1*x3*y2 + \
        2*x1*x3*y3 - 4*x2**2*y1 - 4*x2**2*y3 + 4*x2*x3*y1 + 4*x2*x3*y2 - 2*x3**2*y1
    c = y1**2 - 4*y1*y2 + 2*y1*y3 + 4*y2**2 - 4*y2*y3 + y3**2
    d = x1**2 - 4*x1*x2 + 2*x1*x3 + 4*x2**2 - 4*x2*x3 + x3**2
    e = -2*x1*y1 + 4*x1*y2 - 2*x1*y3 + 4*x2*y1 - 8*x2*y2 +4*x2*y3 - 2*x3*y1 + 4*x3*y2 - 2*x3*y3

    return a,b,c,d,e

# convert quadratic Bézier curve to quadratic polynomials
cof_x, cof_y, cof_xx, cof_yy, cof_xy = get_coef(x1, x2, x3, y1, y2, y3)


# define the partial derivative of drift to x
D_X = ( cof_x * 1 + cof_xx * 2 * G_1[:,0] + cof_xy * G_1[:,1]).reshape(G_1.shape[0],1)

# define the partial derivative of drift to y
D_Y = ( cof_y * 1 + cof_yy * 2 * G_1[:,1] + cof_xy * G_1[:,0]).reshape(G_1.shape[0],1)

# define the interface points contribution to drift
D_I = ( cof_x*ref_layer_points[:,0] + cof_y*ref_layer_points[:,1] + cof_xx*ref_layer_points[:,0]**2 + \
        cof_yy*ref_layer_points[:,1]**2 + cof_xy * ref_layer_points[:,0] * ref_layer_points[:,1] - \
        cof_x*rest_layer_points[:,0] - cof_y*rest_layer_points[:,1] - cof_xx*rest_layer_points[:,0]**2 - \
        cof_yy*rest_layer_points[:,1]**2 - cof_xy * rest_layer_points[:,0] * rest_layer_points[:,1])\
                                                                .reshape(rest_layer_points.shape[0],1)

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