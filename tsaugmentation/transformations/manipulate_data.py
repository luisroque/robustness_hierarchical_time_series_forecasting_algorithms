from scipy.interpolate import CubicSpline
import numpy as np


class ManipulateData:
    def __init__(self, x, transformation, parameters):
        self.x = np.array(x)
        self.transformation = transformation
        self.orig_steps = np.arange(self.x.shape[0])
        self.sigma = parameters

    def _jitter(self):
        sigma = np.std(self.x, axis=0) / 4 * self.sigma[0]
        return self.x + np.random.normal(
            loc=0.0, scale=sigma, size=(self.x.shape[0], self.x.shape[1])
        )

    def _scaling(self):
        factor = np.random.normal(
            loc=1.0, scale=self.sigma[1], size=(self.x.shape[0], self.x.shape[1])
        )
        return np.squeeze(self.x) * factor

    def _magnitude_warp(self, knot=4):
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma[2], size=(knot + 2, self.x.shape[1])
        )
        warp_steps = np.linspace(0, self.x.shape[0] - 1.0, num=knot + 2)
        warper = np.zeros((self.x.shape[0], self.x.shape[1]))

        for i in range(self.x.shape[1]):
            warper[:, i] = np.array(
                [CubicSpline(warp_steps, random_warps[:, i])(self.orig_steps)]
            )
        ret = self.x * warper
        return ret

    def _time_warp(self, knot=4):
        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma[3], size=(knot + 2, self.x.shape[1])
        )
        warp_steps = np.linspace(0, self.x.shape[0] - 1.0, num=knot + 2)
        time_warp = np.zeros((self.x.shape[0], self.x.shape[1]))
        ret = np.zeros((self.x.shape[0], self.x.shape[1]))

        for i in range(self.x.shape[1]):
            time_warp[:, i] = CubicSpline(warp_steps, warp_steps * random_warps[:, i])(
                self.orig_steps
            )
            ret[:, i] = np.interp(self.orig_steps, time_warp[:, i], self.x[:, i])
        return ret

    def apply_transf(self):
        x_new = getattr(ManipulateData, "_" + self.transformation)(self)
        return x_new
