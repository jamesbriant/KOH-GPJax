import numpy as np
import jax.numpy as jnp

class DataLoader:
    def __init__(
            self, 
            data_field_path: str = 'data/simple_field.csv', 
            data_comp_path: str = 'data/simple_comp.csv'
    ):
        DATAFIELD = np.loadtxt(data_field_path, delimiter=',', dtype=np.float32)
        DATACOMP = np.loadtxt(data_comp_path, delimiter=',', dtype=np.float32)

        self.xf = np.reshape(DATAFIELD[:, 0], (-1, 1))
        self.xc = np.reshape(DATACOMP[:, 0], (-1,1))
        self.tc = np.reshape(DATACOMP[:, 1], (-1,1))
        self.yf = np.reshape(DATAFIELD[:, 1], (-1,1))
        self.yc = np.reshape(DATACOMP[:, 2], (-1,1))

        #Standardize full response using mean and std of yc
        self.yc_mean = np.mean(self.yc)
        # self.yc_std = np.std(self.yc, ddof=1) #estimate is now unbiased
        # self.x_min = min(self.xf.min(), self.xc.min())
        # self.x_max = max(self.xf.max(), self.xc.max())
        self.t_min = self.tc.min()
        self.t_max = self.tc.max()

        self.yc_centered = self.yc - self.yc_mean
        self.yf_centered = self.yf - self.yc_mean

        # self.xf_normalized = (self.xf - self.x_min)/(self.x_max - self.x_min)
        # self.xc_normalized = (self.xc - self.x_min)/(self.x_max - self.x_min)
        # tc_normalized = np.zeros_like(tc)
        # for k in range(tc.shape[1]):
        #     tc_normalized[:, k] = (tc[:, k] - np.min(tc[:, k]))/(np.max(tc[:, k]) - np.min(tc[:, k]))
        self.tc_normalized = (self.tc - self.t_min)/(self.t_max - self.t_min)
        # self.yc_standardized = (self.yc - self.yc_mean)/self.yc_std
        # self.yf_standardized = (self.yf - self.yc_mean)/self.yc_std

        # self.x_stack = jnp.vstack((self.xf_normalized, self.xc_normalized), dtype=np.float64)
        # self.y = jnp.vstack((self.yf_standardized, self.yc_standardized), dtype=np.float64)
        
        self.x_stack = jnp.vstack((self.xf, self.xc), dtype=np.float64)
        self.y = jnp.vstack((self.yf_centered, self.yc_centered), dtype=np.float64)

    def get_data(self):
        return self.x_stack, self.tc_normalized, self.y
        # return self.x_stack, self.tc, self.y
    
    # def transform_x(self, x):
    #     return (x - self.x_min)/(self.x_max - self.x_min)
    
    # def inverse_transform_x(self, x):
    #     return x*(self.x_max - self.x_min) + self.x_min
    
    # def transform_t(self, t):
    #     return (t - self.t_min)/(self.t_max - self.t_min)
    
    # def inverse_transform_t(self, t):
    #     return t*(self.t_max - self.t_min) + self.t_min
    
    # def transform_y(self, y):
    #     return (y - self.yc_mean)/self.yc_std

    def transform_y(self, y):
        return y - self.yc_mean
    
    # def inverse_transform_y(self, y):
    #     return y*self.yc_std + self.yc_mean

    def inverse_transform_y(self, y):
        return y + self.yc_mean
    
    # def transform_y_cov(self, cov):
    #     return cov/(self.yc_std**2)
    
    # def inverse_transform_y_cov(self, cov):
    #     return cov*(self.yc_std**2)