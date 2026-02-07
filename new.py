import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple

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
def gradient_descent_step(pos: np.ndarray, lr: float = 0.03) -> np.ndarray:
    return pos - lr * gradient(*pos)

def adam_step(pos: np.ndarray, m: np.ndarray, v: np.ndarray, t: int,
              lr: float = 0.08, beta1: float = 0.9, beta2: float = 0.999,
              eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = gradient(*pos)
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*(g**2)
    m_hat = m / (1-beta1**t)
    v_hat = v / (1-beta2**t)
    return pos - lr*m_hat/(np.sqrt(v_hat)+eps), m, v

# --------------------------
# Simulation utilities
# --------------------------
def simulate_optimizers(start_pos: np.ndarray = None, steps: int = 150,
                        lr_gd: float = 0.03, lr_adam: float = 0.08) -> Dict[str, List[np.ndarray]]:
    if start_pos is None:
        start_pos = np.array([2.5, 2.5])

    optimizers = {
        'GD': {'pos': start_pos.copy()},
        'Adam': {'pos': start_pos.copy(), 'm': np.zeros(2), 'v': np.zeros(2)}
    }

    paths = {key: [opt['pos'].copy()] for key, opt in optimizers.items()}

    for t in range(1, steps+1):
        for key, opt in optimizers.items():
            pos = opt['pos']

            if key == 'GD':
                pos = gradient_descent_step(pos, lr=lr_gd)
                opt['pos'] = pos
            elif key == 'Adam':
                pos, m, v = adam_step(pos, opt['m'], opt['v'], t=t, lr=lr_adam)
                opt['pos'] = pos
                opt['m'] = m
                opt['v'] = v

            paths[key].append(opt['pos'].copy())

    return paths

def create_surface(xmin: float = -3, xmax: float = 3, res: int = 200):
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(xmin, xmax, res)
    X, Y = np.meshgrid(x, y)
    Z = loss(X, Y)
    return X, Y, Z

def animate_paths(paths: Dict[str, List[np.ndarray]], X, Y, Z,
                  interval: int = 200, figsize: Tuple[int,int] = (9,7)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    colors = {'GD':'r', 'Adam':'b'}

    # Prepare dots and trails
    dots = {}
    trails = {}
    max_frames = max(len(p) for p in paths.values())

    for key in paths.keys():
        z = loss(*paths[key][0])
        dots[key], = ax.plot([paths[key][0][0]], [paths[key][0][1]], [z],
                             'o', color=colors.get(key,'k'), markersize=8, label=key)
        trails[key], = ax.plot([paths[key][0][0]], [paths[key][0][1]], [z],
                               '--', color=colors.get(key,'k'), linewidth=2)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(0, float(Z.max()))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Loss')
    ax.set_title("Gradient Descent vs Adam (Simulation)")
    ax.legend()

    def update(frame):
        for key in paths.keys():
            idx = min(frame, len(paths[key]) - 1)
            pos = paths[key][idx]
            z = loss(*pos)
            dots[key].set_data([pos[0]], [pos[1]])
            dots[key].set_3d_properties([z])

            path_array = np.array(paths[key][:idx+1])
            z_path = np.array([loss(px, py) for px, py in path_array])
            trails[key].set_data(path_array[:,0], path_array[:,1])
            trails[key].set_3d_properties(z_path)

        return list(dots.values()) + list(trails.values())

    anim = FuncAnimation(fig, update, frames=max_frames, interval=interval, blit=False)
    return fig, anim


if __name__ == "__main__":
    # simple demo when run as a script
    X, Y, Z = create_surface()
    paths = simulate_optimizers(steps=150)
    fig, anim = animate_paths(paths, X, Y, Z, interval=200)
    plt.show()
