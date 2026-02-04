import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# Loss function
# --------------------------
def loss(x, y):
    return x**2 + y**2 + np.sin(3*x)*np.sin(3*y)

def gradient(x, y):
    dx = 2*x + 3*np.cos(3*x)*np.sin(3*y)
    dy = 2*y + 3*np.sin(3*x)*np.cos(3*y)
    return np.array([dx, dy])

# --------------------------
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

# --------------------------
# Initialize positions
# --------------------------
start_pos = np.array([2.5, 2.5])

optimizers = {
    'GD': {'pos': start_pos.copy()},
    'Adam': {'pos': start_pos.copy(), 'm': np.zeros(2), 'v': np.zeros(2)}
}

colors = {'GD':'r', 'Adam':'b'}
paths = {key: [opt['pos'].copy()] for key, opt in optimizers.items()}

# --------------------------
# 3D Surface plot
# --------------------------
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = loss(X, Y)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Dots and trails
dots = {}
trails = {}
for key in paths.keys():
    z = loss(*paths[key][-1])
    dots[key], = ax.plot([paths[key][-1][0]], [paths[key][-1][1]], [z],
                         'o', color=colors[key], markersize=8, label=key)
    trails[key], = ax.plot([paths[key][-1][0]], [paths[key][-1][1]], [z],
                           '--', color=colors[key], linewidth=2)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Loss')
ax.set_title("Gradient Descent vs Adam (Interactive)")
ax.legend()

# --------------------------
# Update function
# --------------------------
def update(frame):
    for key, opt in optimizers.items():
        pos = opt['pos']

        if key == 'GD':
            pos = gradient_descent_step(pos)
            opt['pos'] = pos
        elif key == 'Adam':
            pos, m, v = adam_step(pos, opt['m'], opt['v'], t=frame+1)
            opt['pos'] = pos
            opt['m'] = m
            opt['v'] = v

        paths[key].append(opt['pos'].copy())

        # Update dot
        z = loss(*opt['pos'])
        dots[key].set_data([opt['pos'][0]], [opt['pos'][1]])
        dots[key].set_3d_properties([z])

        # Update trail
        path_array = np.array(paths[key])
        z_path = np.array([loss(px, py) for px, py in path_array])
        trails[key].set_data(path_array[:,0], path_array[:,1])
        trails[key].set_3d_properties(z_path)

    return list(dots.values()) + list(trails.values())

# --------------------------
# Run animation
# --------------------------
anim = FuncAnimation(fig, update, frames=150, interval=400, blit=False)  # slower, clear

# --------------------------
# Interactive view: rotate and zoom with mouse
plt.show()
