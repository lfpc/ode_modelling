'''Script for solving the equations using the Runge-Kutta method.'''

import numpy as np

def runge_kutta(f, y0, t0, tf, dt):
    '''Runge-Kutta method for solving ordinary differential equations.

    Parameters:
    f : function
        The function that defines the system of equations. It should take two arguments: time and state.
    y0 : array-like
        The initial state of the system.
    t0 : float
        The initial time.
    tf : float
        The final time.
    dt : float
        The time step.

    Returns:
    t : numpy array
        The array of time points.
    y : numpy array
        The array of states corresponding to each time point.
    '''
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + dt/2, y[i-1] + dt/2 * k1)
        k3 = f(t[i-1] + dt/2, y[i-1] + dt/2 * k2)
        k4 = f(t[i-1] + dt, y[i-1] + dt * k3)
        y[i] = y[i-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y


