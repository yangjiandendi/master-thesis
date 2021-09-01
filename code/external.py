import numpy as np
import matplotlib.pyplot as plt

class external_kriging():

    def __init__(self,G_1,G_1_o,layer1,layer2,a_T,con_p):

        self.G_1 = G_1
        self.G_1_o =G_1_o
        self.layer1 = layer1
        self.layer2 = layer2
        self.a_T = a_T
        self.con_p = con_p
        self.c_o_T = a_T**2/14/3
        self.number_of_points_per_surface = np.array([layer1.shape[0],layer2.shape[0]])
        self.ref_layer_points = np.repeat(np.stack([layer1[-1],layer2[-1]],axis = 0),repeats=self.number_of_points_per_surface-1,axis = 0)
        self.rest_layer_points = np.concatenate([layer1[0:-1],layer2[0:-1]],axis = 0)
        self.G_1_tiled = np.tile(G_1,[2,1])

    def squared_euclidean_distance(self,x_1,x_2):
        sqd = np.sqrt(np.reshape(np.sum(x_1**2,1),newshape =(x_1.shape[0],1))+\
        np.reshape(np.sum(x_2**2,1),newshape =(1,x_2.shape[0]))-\
        2*(x_1@x_2.T))
        return sqd

    def cartesian_dist(self,x_1,x_2):
        return np.concatenate([
            np.tile(x_1[:,0] - np.reshape(x_2[:,0],[x_2.shape[0],1]),[1,2]),
            np.tile(x_1[:,1] - np.reshape(x_2[:,1],[x_2.shape[0],1]),[1,2])],axis = 0) 

    def cartesian_dist_no_tile(self,x_1,x_2):
        return np.concatenate([
            np.transpose((x_1[:,0] - np.reshape(x_2[:,0],[x_2.shape[0],1]))),
            np.transpose((x_1[:,1] - np.reshape(x_2[:,1],[x_2.shape[0],1])))],axis = 0) 

    def perpendicularity(self,G_1):
        a = np.concatenate([np.ones([G_1.shape[0],G_1.shape[0]]),np.zeros([G_1.shape[0],G_1.shape[0]])],axis = 1)
        b = np.concatenate([np.zeros([G_1.shape[0],G_1.shape[0]]),np.ones([G_1.shape[0],G_1.shape[0]])],axis = 1)
        return np.concatenate([a,b],axis = 0)

    def cov_gradients(self):

        c_o_T = self.a_T**2/14/3

        G_1_tiled = np.tile(self.G_1,[2,1])

        h_u = self.cartesian_dist(self.G_1,self.G_1)
        h_v = h_u.T

        perpendicularity_matrix = self.perpendicularity(self.G_1)

        dist_tiled = self.squared_euclidean_distance(G_1_tiled,G_1_tiled)

        dist_tiled = dist_tiled + np.eye(dist_tiled.shape[0])
    
        condition1 = 0
        a = (h_u*h_v)
        b = dist_tiled**2

        t1 =  np.divide(a, b, out=np.zeros_like(a),casting='unsafe', where=b!=0)
        t2 = np.where(dist_tiled < self.a_T,(-c_o_T*((-14/self.a_T**2)+
                    105/4*dist_tiled/self.a_T**3 -
                    35/2 * dist_tiled**3 / self.a_T **5 +
                    21 /4 * dist_tiled**5/self.a_T**7)),0)+\
            np.where(dist_tiled < self.a_T,(c_o_T * 7 * (9 * dist_tiled ** 5 -
                        20 * self.a_T ** 2 * dist_tiled ** 3 +
                        15 * self.a_T ** 4 * dist_tiled -
                        4 * self.a_T ** 5) / (2 * self.a_T ** 7)),0)

        # when we do the covariance of Gx with Gx, Gy with Gy and so on, quation B9 in Gempy paper
        t3 = perpendicularity_matrix * \
            np.where(dist_tiled < self.a_T,(c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * dist_tiled / self.a_T ** 3 -
                    35 / 2 * dist_tiled ** 3 / self.a_T ** 5 +
                    21 / 4 * dist_tiled ** 5 / self.a_T ** 7)),0)
        t4 = 1/3*np.eye(dist_tiled.shape[0])

        condition2 = t1 * t2 - t3 + t4

        C_G = np.where(dist_tiled==0, condition1, condition2) ## adding nugget effect
        return C_G
    
    def cov_interface(self):

        sed_rest_rest = self.squared_euclidean_distance(self.rest_layer_points,self.rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distance(self.ref_layer_points,self.rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distance(self.rest_layer_points,self.ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distance(self.ref_layer_points,self.ref_layer_points)
        C_I = self.c_o_T*(\
                    (np.where(sed_rest_rest < self.a_T, (1 - 7 * (sed_rest_rest / self.a_T) ** 2 +\
                    35 / 4 * (sed_rest_rest / self.a_T) ** 3 -\
                    7 / 2 * (sed_rest_rest / self.a_T) ** 5 +\
                    3 / 4 * (sed_rest_rest / self.a_T) ** 7),0)) -\
                    (np.where(sed_ref_rest < self.a_T, (1 - 7 * (sed_ref_rest / self.a_T) ** 2 +\
                    35 / 4 * (sed_ref_rest / self.a_T) ** 3 -\
                    7 / 2 * (sed_ref_rest / self.a_T) ** 5 +\
                    3 / 4 * (sed_ref_rest / self.a_T) ** 7),0)) -\
                    (np.where(sed_rest_ref < self.a_T, (1 - 7 * (sed_rest_ref / self.a_T) ** 2 +\
                    35 / 4 * (sed_rest_ref / self.a_T) ** 3 -\
                    7 / 2 * (sed_rest_ref / self.a_T) ** 5 +\
                    3 / 4 * (sed_rest_ref / self.a_T) ** 7),0)) +\
                    (np.where(sed_ref_ref < self.a_T, (1 - 7 * (sed_ref_ref / self.a_T) ** 2 +\
                    35 / 4 * (sed_ref_ref / self.a_T) ** 3 -\
                    7 / 2 * (sed_ref_ref / self.a_T) ** 5 +\
                    3 / 4 * (sed_ref_ref / self.a_T) ** 7),0)))
        
        return C_I

    def cov_interface_gradients(self):

        hu_rest = self.cartesian_dist_no_tile(self.G_1,self.rest_layer_points)
        hu_ref = self.cartesian_dist_no_tile(self.G_1,self.ref_layer_points)

        sed_dips_rest = self.squared_euclidean_distance(self.G_1_tiled,self.rest_layer_points)
        sed_dips_ref = self.squared_euclidean_distance(self.G_1_tiled,self.ref_layer_points)

        C_GI = hu_rest*np.where(sed_dips_rest < self.a_T,(- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_rest / self.a_T ** 3 -
                                    35 / 2 * sed_dips_rest ** 3 / self.a_T ** 5 +
                                    21 / 4 * sed_dips_rest ** 5 / self.a_T ** 7)),0)-\
        hu_ref*np.where(sed_dips_ref < self.a_T,(-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_ref / self.a_T ** 3 -
                                    35 / 2 * sed_dips_ref ** 3 / self.a_T ** 5 +
                                    21 / 4 * sed_dips_ref ** 5 / self.a_T ** 7)),0)
        return C_GI

    def get_coef(self):
        """
        Convert quadratic Bézier curve to quadratic polynomials

        Args:
            x1, x2, x3: x coodinates of control points
            y1, y2, y3: y coodinates of control points
            
        Returns:
            a, b, c, d, e: coefficient of quadratic polynomials
        """
        x1 = self.con_p[0,0]
        x2 = self.con_p[1,0]
        x3 = self.con_p[2,0]
        y1 = self.con_p[0,1]
        y2 = self.con_p[1,1]
        y3 = self.con_p[2,1]

        a = 2*x1*y1*y3 - 4*x1*y2**2 + 4*x1*y2*y3 -2*x1*y3**2 + 4*x2*y1*y2 - \
            8*x2*y1*y3 + 4*x2*y2*y3 - 2*x3*y1**2 + 4*x3*y1*y2 + 2*x3*y1*y3 - 4*x3*y2**2 
        b = -2*x1**2*y3 + 4*x1*x2*y2 + 4*x1*x2*y3 + 2*x1*x3*y1 - 8*x1*x3*y2 + \
            2*x1*x3*y3 - 4*x2**2*y1 - 4*x2**2*y3 + 4*x2*x3*y1 + 4*x2*x3*y2 - 2*x3**2*y1
        c = y1**2 - 4*y1*y2 + 2*y1*y3 + 4*y2**2 - 4*y2*y3 + y3**2
        d = x1**2 - 4*x1*x2 + 2*x1*x3 + 4*x2**2 - 4*x2*x3 + x3**2
        e = -2*x1*y1 + 4*x1*y2 - 2*x1*y3 + 4*x2*y1 - 8*x2*y2 +4*x2*y3 - 2*x3*y1 + 4*x3*y2 - 2*x3*y3

        return a,b,c,d,e

    def safe_arange(self,start, stop, step):
        return step * np.arange(start / step, stop / step)


    def k_matrix(self):

        C_G = self.cov_gradients()
        C_I = self.cov_interface()
        C_GI = self.cov_interface_gradients()
        C_IG = C_GI.T
        K = np.concatenate([np.concatenate([C_G,C_GI],axis = 1),np.concatenate([C_IG,C_I],axis = 1)],axis = 0)
        return K

    def get_grid(self):
        xx = self.safe_arange(0,20,0.1)
        yy = self.safe_arange(0,10,0.1)
        XX,YY = np.meshgrid(xx,yy)
        X = (np.reshape(XX,[-1])).T
        Y = (np.reshape(YY,[-1])).T
        grid = np.stack([X,Y],axis = 1)
        return grid

    def interpo_va(self):
        grid = self.get_grid()
        hu_Simpoints = self.cartesian_dist_no_tile(self.G_1,grid)
        sed_dips_SimPoint = self.squared_euclidean_distance(self.G_1_tiled,grid)
        sed_rest_SimPoint = self.squared_euclidean_distance(self.rest_layer_points,grid)
        sed_ref_SimPoint = self.squared_euclidean_distance(self.ref_layer_points,grid)
        return hu_Simpoints,sed_dips_SimPoint,sed_rest_SimPoint,sed_ref_SimPoint
    
    def D_matrix(self):
        # convert quadratic Bézier curve to quadratic polynomials
        cof_x, cof_y, cof_xx, cof_yy, cof_xy = self.get_coef()


        # define the partial derivative of drift to x
        D_X = ( cof_x * 1 + cof_xx * 2 * self.G_1[:,0] + cof_xy * self.G_1[:,1]).reshape(self.G_1.shape[0],1)

        # define the partial derivative of drift to y
        D_Y = ( cof_y * 1 + cof_yy * 2 * self.G_1[:,1] + cof_xy * self.G_1[:,0]).reshape(self.G_1.shape[0],1)

        # define the interface points contribution to drift
        D_I = ( cof_x*self.ref_layer_points[:,0] + cof_y*self.ref_layer_points[:,1] + cof_xx*self.ref_layer_points[:,0]**2 + \
                cof_yy*self.ref_layer_points[:,1]**2 + cof_xy * self.ref_layer_points[:,0] * self.ref_layer_points[:,1] - \
                cof_x*self.rest_layer_points[:,0] - cof_y*self.rest_layer_points[:,1] - cof_xx*self.rest_layer_points[:,0]**2 - \
                cof_yy*self.rest_layer_points[:,1]**2 - cof_xy * self.rest_layer_points[:,0] * self.rest_layer_points[:,1])\
                                                                        .reshape(self.rest_layer_points.shape[0],1)

        # build the dirft matrix
        D = np.vstack([D_X , D_Y , D_I])
        D_T = D.T
        return D,D_T

    # build zero matrix


    def new_K(self):
        # concatenate drift matrix and kriging matrix
        K = self.k_matrix()
        D,D_T = self.D_matrix()
        zero_matrix = np.zeros([1,1])
        K_D = np.concatenate([np.concatenate([K,D],axis = 1),np.concatenate([D_T,zero_matrix],axis = 1)],axis = 0)
        return K_D


    def b_matrix(self):
    # build right side matrix of cokriging system
        K_D = self.new_K()
        bk = np.concatenate([self.G_1_o[:,0],self.G_1_o[:,1],np.zeros(K_D.shape[0]-self.G_1.shape[0]*2)],axis = 0)
        bk = np.reshape(bk,newshape = [bk.shape[0],1])
        return bk
    def get_w(self):
        # solve kriging weight
        K_D = self.new_K()
        bk = self.b_matrix()
        w = np.linalg.lstsq(K_D,bk)[0]
        return w

    def get_intp(self):
        w = self.get_w()
        hu_Simpoints,sed_dips_SimPoint,sed_rest_SimPoint,sed_ref_SimPoint = self.interpo_va()
        grid = self.get_grid()
        # gradient contribution
        sigma_0_grad = w[:self.G_1.shape[0]*2] * (-hu_Simpoints*(sed_dips_SimPoint < self.a_T)*(- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 - 35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 + 21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7)))

        sigma_0_grad = np.sum(sigma_0_grad,axis=0)
        # surface point contribution
        sigma_0_interf = -w[self.G_1.shape[0]*2:-1]*(self.c_o_T  * ((sed_rest_SimPoint < self.a_T)*(1 - 7 * (sed_rest_SimPoint / self.a_T) ** 2 +
                    35 / 4 * (sed_rest_SimPoint / self.a_T) ** 3 -
                    7 / 2 * (sed_rest_SimPoint / self.a_T) ** 5 +
                    3 / 4 * (sed_rest_SimPoint / self.a_T) ** 7) -
                    (sed_ref_SimPoint < self.a_T)*(1 - 7 * (sed_ref_SimPoint / self.a_T) ** 2 +
                    35 / 4 * (sed_ref_SimPoint / self.a_T) ** 3 -
                    7 / 2 * (sed_ref_SimPoint / self.a_T) ** 5 +
                    3 / 4 * (sed_ref_SimPoint / self.a_T) ** 7)))
        sigma_0_interf = np.sum(sigma_0_interf,axis = 0)

        # 2nd order drift contribution
        cof_x, cof_y, cof_xx, cof_yy, cof_xy = self.get_coef()

        sigma_0_2nd_drift_1 = grid[:,0]* (w[-1]).T * cof_x

        sigma_0_2nd_drift_2 = grid[:,1]* (w[-1]).T * cof_y

        sigma_0_2nd_drift_3 = grid[:,0]**2 * (w[-1]).T * cof_xx

        sigma_0_2nd_drift_4 = grid[:,1]**2 * (w[-1]).T * cof_yy

        sigma_0_2nd_drift_5 = grid[:,1] * grid[:,0] * (w[-1]).T * cof_xy




        sigma_0_2nd_drift = sigma_0_2nd_drift_1 + sigma_0_2nd_drift_2 + sigma_0_2nd_drift_3 + sigma_0_2nd_drift_4 + sigma_0_2nd_drift_5


        interpolate_result = sigma_0_grad+sigma_0_interf+sigma_0_2nd_drift

        intp = np.reshape(interpolate_result,[100,200]) # reshape the result to matrix shape

        return intp

    def plot_value(self):
        
        intp = self.get_intp()
        #plt.contour(XX,YY,intp,50)
        xx = self.safe_arange(0,20,0.1)
        yy = self.safe_arange(0,10,0.1)
        XX,YY = np.meshgrid(xx,yy)

        layer_lvl=np.concatenate([intp[np.where((XX==self.layer1[0,0])&(YY==self.layer1[0,1]))],intp[np.where((XX==self.layer2[0,0])&(YY==self.layer2[0,1]))]])

        plt.contourf(XX,YY,intp,levels=layer_lvl)
        plt.pcolor(XX,YY,intp, alpha=0.3, shading='auto')

        #plt.colorbar()
        plt.plot(self.layer1[:,0], self.layer1[:,1], 'ro')
        plt.plot(self.layer2[:,0], self.layer2[:,1], 'bo')

        plt.plot(self.G_1[:,0], self.G_1[:,1], 'go')


        plt.quiver([self.G_1[:,0]],[self.G_1[:,1]],self.G_1_o[:,0],self.G_1_o[:,1],color='r')


        mm=np.array([])
        nn=np.array([])
        for t in np.arange(0,1,0.01):
            m = np.array([self.con_p[0,0]*t**2 + self.con_p[1,0]*t*(1-t)*2 + self.con_p[2,0]*(1-t)**2])
            n = np.array([self.con_p[0,1]*t**2 + self.con_p[1,1]*t*(1-t)*2 + self.con_p[2,1]*(1-t)**2])
            mm = np.append(mm,m)
            nn = np.append(nn,n)
        plt.plot(mm,nn,color='k')

        plt.plot(self.con_p[:,0],self.con_p[:,1],'ko')
        plt.plot(self.con_p[:,0],self.con_p[:,1],'--')
        plt.show()


    

    
