from scipy.interpolate import CubicSpline

class manipulate_data():
    
    def __init__(self, x, transformation):
        self.x = x
        self.transformation = transformation

    def _jitter(self, sigma=0.2):
        return x + np.random.normal(loc=0., scale=sigma, size=self.x.shape)

    def _scaling(self, sigma=0.2):
        factor = np.random.normal(loc=1., scale=sigma, size=(self.x.shape[0],self.x.shape[2]))
        return np.multiply(self.x, factor[:,np.newaxis,:])

    def _magnitude_warp(self, sigma=0.25, knot=4):
        orig_steps = np.arange(self.x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(self.x.shape[0], knot+2))
        warp_steps = ((np.linspace(0, self.x.shape[1]-1., num=knot+2)))
        ret = np.zeros_like(self.x)
        for i, pat in enumerate(self.x):
            warper = np.array([CubicSpline(warp_steps, random_warps[i,:])(orig_steps)])
            ret[i] = pat * warper

        return ret

    def _time_warp(self, sigma=0.025, knot=4):
        orig_steps = np.arange(self.x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(self.x.shape[0], knot+2))
        warp_steps = ((np.linspace(0, self.x.shape[1]-1., num=knot+2)))

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            time_warp = CubicSpline(warp_steps, warp_steps * random_warps[i,:])(orig_steps)
            scale = (self.x.shape[1]-1)/time_warp[-1]
            ret[i,:] = np.interp(orig_steps, np.clip(scale*time_warp, 0, self.x.shape[1]-1), pat).T
        return ret  
    
    def apply_transf(self):
        x_new = getattr(manipulate_data, '_' + self.transformation)(self)
        return x_new