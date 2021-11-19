from scipy.interpolate import CubicSpline
import numpy as np


class ManipulateData:
    
    def __init__(self, x, transformation, parameters):
        self.x = np.array(x)
        self.transformation = transformation
        self.orig_steps = np.arange(self.x.shape[0])
        self.sigma = parameters

    def _jitter(self):
        sigma = np.std(self.x, axis=0)/4*self.sigma
        return np.squeeze(self.x) + np.random.normal(loc=0., scale=sigma, size=self.x.shape[0])

    def _scaling(self):
        factor = np.random.normal(loc=1., scale=self.sigma, size=(self.x.shape[0]))
        return np.squeeze(self.x) * factor

    def _magnitude_warp(self, knot=4):
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=knot+2)
        warp_steps = (np.linspace(0, self.x.shape[0]-1., num=knot+2))
        warper = np.array([CubicSpline(warp_steps, random_warps)(self.orig_steps)])
        ret = self.x * warper.reshape(-1, 1)
        return np.squeeze(ret)

    def _time_warp(self, knot=4):
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=knot+2)
        warp_steps = (np.linspace(0, self.x.shape[0]-1., num=knot+2))

        time_warp = CubicSpline(warp_steps, warp_steps * random_warps)(self.orig_steps)
        scale = (self.x.shape[0]-1)/time_warp[-1]
        ret = np.interp(self.orig_steps, np.clip(scale*time_warp, 0, self.x.shape[0]-1), np.squeeze(self.x))
        return ret
    
    def apply_transf(self):
        x_new = getattr(ManipulateData, '_' + self.transformation)(self)
        return x_new
