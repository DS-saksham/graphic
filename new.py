import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Loss function
# --------------------------
def loss(x, y):
    return x**2 + y**2 + np.sin(3*x)*np.sin(3*y)

def gradient(x, y):
    dx = 2*x + 3*np.cos(3*x)*np.sin(3*y)
    dy = 2*y + 3*np.sin(3*x)*np.cos(3*y)
    return np.array([dx, dy])

# Optimizer steps
# --------------------------
def gradient_descent_step(pos, lr=0.03):  # slower GD
    return pos - lr * gradient(*pos)

def adam_step(pos, m, v, t, lr=0.08, beta1=0.9, beta2=0.999, eps=1e-8):  # faster Adam
    g = gradient(*pos)
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*(g**2)
    m_hat = m / (1-beta1**t)
    v_hat = v / (1-beta2**t)
    return pos - lr*m_hat/(np.sqrt(v_hat)+eps), m, v