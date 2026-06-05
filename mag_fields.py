import pickle
import numpy as np

class CustomMagneticField:
    def __init__(self, field_map):
        self.initialize_grid(field_map)

    def initialize_grid(self, field_map):
        with open(field_map, 'rb') as f:
            field_map = pickle.load(f)
        points = np.array(field_map['points'])
        self.x_min, self.x_max = points[:, 0].min(), points[:, 0].max()
        self.y_min, self.y_max = points[:, 1].min(), points[:, 1].max()
        self.z_min, self.z_max = points[:, 2].min(), points[:, 2].max()
        
        self.dx = np.diff(np.unique(points[:,0]))[0]
        self.dy = np.diff(np.unique(points[:,1]))[0]
        self.dz = np.diff(np.unique(points[:,2]))[0]
        
        self.dx_inv = 1.0 / self.dx
        self.dy_inv = 1.0 / self.dy
        self.dz_inv = 1.0 / self.dz
        
        self.fields = np.array(field_map['B'])
        self.nx = int(round((self.x_max - self.x_min) * self.dx_inv)) + 1
        self.ny = int(round((self.y_max - self.y_min) * self.dy_inv)) + 1
        self.nz = int(round((self.z_max - self.z_min) * self.dz_inv)) + 1

    def __call__(self, x, y, z):
        if abs(x) > self.x_max or abs(y) > self.y_max or abs(z) > self.z_max:
            return np.array([0.0, 0.0, 0.0])
        
        quadrant = 0
        if x >= 0 and y >= 0:
            quadrant = 1
        elif x < 0 and y >= 0:
            quadrant = 2
        elif x < 0 and y < 0:
            quadrant = 3
        elif x >= 0 and y < 0:
            quadrant = 4
        
        symmetric_point = np.array([x, y, z])
        if quadrant == 2:
            symmetric_point[0] *= -1
        elif quadrant == 3:
            symmetric_point[0] *= -1
            symmetric_point[1] *= -1
        elif quadrant == 4:
            symmetric_point[1] *= -1

        i = int(round((symmetric_point[0] - self.x_min) * self.dx_inv))
        j = int(round((symmetric_point[1] - self.y_min) * self.dy_inv))
        k = int(round((symmetric_point[2] - self.z_min) * self.dz_inv))

        if i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz:
            return np.array([0.0, 0.0, 0.0])

        idx = j * (self.nx * self.nz) + i * self.nz + k
        Bfield = self.fields[idx]

        if quadrant == 2 or quadrant == 4:
            Bfield[0] = -Bfield[0]
        if quadrant == 3 or quadrant == 4:
            Bfield[2] = -Bfield[2]

        return Bfield

class UniformMagneticField:
    def __init__(self, Bx = 0.0, By = 0.0, Bz = 0.0) -> None:
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
    def __call__(self, pos) -> np.ndarray:
        return np.array([self.Bx, self.By, self.Bz]).reshape(pos.shape[0], 3)
    
class ToyMagneticField:
    @staticmethod
    def __call__(x: float, y: float, z: float) -> np.ndarray:
        B = 1
        if z < 10:
            return np.array([0.0, 0.0, 0.0])
        elif  z< 20:
            return np.array([0.0, 1.0, 0.0])*B
        elif z < 30:
            return np.array([0.0, -1.0, 0.0])*B
        elif z < 40:
            return np.array([1.0, 0.0, 0.0])*B
        elif z < 50:
            return np.array([-1.0, 0.0, 0.0])*B
        elif z < 60:
            return np.array([0.0, -1.0, 0.0])*B
        elif z < 70:
            return np.array([0.0, 1.0, 0.0])*B
        else:
            return np.array([0.0, 0.0, 0.0])
