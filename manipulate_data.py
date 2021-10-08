from scipy.interpolate import CubicSpline

class manipulate_data():
    
    def __init__(self, x):
        self.x = x

    def jitter(self, sigma=0.2):
        return x + np.random.normal(loc=0., scale=sigma, size=self.x.shape)

    def scaling(self, sigma=0.2):
        factor = np.random.normal(loc=1., scale=sigma, size=(self.x.shape[0],self.x.shape[2]))
        return np.multiply(self.x, factor[:,np.newaxis,:])

    def magnitude_warp(self, sigma=0.2, knot=4):
        orig_steps = np.arange(self.x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(self.x.shape[0], knot+2, self.x.shape[2]))
        warp_steps = (np.ones((self.x.shape[2],1))*(np.linspace(0, self.x.shape[1]-1., num=knot+2))).T
        ret = np.zeros_like(self.x)
        for i, pat in enumerate(self.x):
            warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(self.x.shape[2])]).T
            ret[i] = pat * warper

        return ret

    def time_warp(self, sigma=0.1, knot=4):
        orig_steps = np.arange(self.x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(self.x.shape[0], knot+2, self.x.shape[2]))
        warp_steps = (np.ones((self.x.shape[2],1))*(np.linspace(0, self.x.shape[1]-1., num=knot+2))).T

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (self.x.shape[1]-1)/time_warp[-1]
                ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, self.x.shape[1]-1), pat[:,dim]).T
        return ret  