import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

######################### FUNCTIONs
def l2p(x, y, z):
    de = z + 1
    return x / de, y / de 

def l2k(x, y, z):
    return x / z, y / z 

def k2l_gamma(x, y):
    gamma = 1 / np.sqrt(1 - x ** 2 - y ** 2)
    return gamma

def p2l(x, y):
    gamma =1 - x ** 2 - y ** 2
    return 2 * x / gamma, 2 * y / gamma, (1 + x ** 2 + y ** 2) / gamma

def line(a, b, c, bound=1, steps=200, mode='circle', balance=0.):
    delta = (b * c)**2 - (b**2 + a**2)*(c**2 - bound * a**2)
    if delta <= 0:
            return None
    low = (-(b*c) - np.sqrt(delta)) / (b**2 + a**2)
    high = (-(b*c) + np.sqrt(delta)) / (b**2 + a**2)
    y = np.linspace(low + (high - low) * balance, high, steps)
    x = -(b*y + c) / a
    # print(y)
    return x, y

def disk_mesh(r, steps=200):
    angle = np.linspace(0, 360, steps)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def ein_midpoint_k(x, y, gamma):
     de =  np.sum(gamma) 
     return np.sum(x * gamma) / de, np.sum(y * gamma) / de

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def line_plot(ax, x, y, z, color):
    if z is not None: 
        for i in range(x.shape[0] - 1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], c=color[i])
    else:
        for i in range(x.shape[0] - 1):
            ax.plot(x[i:i+2], y[i:i+2], c=color[i])

################### PLOTTING 
x = np.linspace(-3, 3, 100)
y = x.copy()
mesh = np.stack(np.meshgrid(x, y), axis=2)
# Curvature of 1
zz = np.sqrt(np.linalg.norm(mesh, axis=2) ** 2 + 1)
mesh = np.concatenate([mesh, zz[:, :, None]], axis=2)

# a, b, c = 1, 2, -1
ky, kx = line(1, -1, 0, bound=0.9, balance=0.)
kx = sigmoid(kx)
ky = sigmoid(ky)
# Lorentz
gamma = k2l_gamma(kx, ky)



lx, ly, lz = kx * gamma, ky * gamma, 1 * gamma
cmap = np.interp(lz, (lz.min(), lz.max()), (1, 0)) 
# Line curvature color
line_map = np.stack([cmap, np.zeros_like(cmap),
                     np.zeros_like(cmap)], axis=1)

# Poincare
px, py = l2p(lx, ly, lz)

# Midpoint 
mk_x, mk_y = ein_midpoint_k(kx[[0, -1]], ky[[0, -1]], gamma[[0, -1]])
m_gamma = k2l_gamma(mk_x, mk_y)
ml_x, ml_y, ml_z = mk_x * m_gamma, mk_y * m_gamma, m_gamma
print(ml_x)
mp_x, mp_y = l2p(ml_x, ml_y, ml_z)



# Plotting 
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
# 2d plotting
ax2 = fig.add_subplot(122)
# Lorentz surface plotting
ax.plot_surface(mesh[:, :, 0], mesh[:, :, 1], zz, color=(0, 1, 0, 0.1))
#Poincare disk
pp = Circle((0, 0), 1, color=[0, 1, 0, 0.1])
ax.add_patch(pp)
art3d.pathpatch_2d_to_3d(pp, z=0, zdir="z")
#Klein disk
pp = Circle((0, 0), 1, color=[0, 1, 0, 0.1])
ax.add_patch(pp)
# add to 2d plot
ax2.add_patch(Circle((0, 0), 1, color=[0, 1, 0, 0.1]))
art3d.pathpatch_2d_to_3d(pp, z=1, zdir="z")

# line plotting
line_plot(ax, kx, ky, np.ones_like(kx), line_map)
line_plot(ax, lx, ly, lz, line_map)
line_plot(ax, px, py, np.zeros_like(px), line_map)
line_plot(ax2, kx, ky, None, line_map)
# line_plot(ax2, px, py, None, np.roll(line_map, -1, axis=1))
# ax.plot(kx, ky, np.ones_like(px), color =  plt.cm.jet(cmap))
# ax.plot(lx, ly, lz, color= plt.cm.jet(cmap))

# Scatter Plotting
ax2.scatter(mk_x, mk_y)
ax.scatter([mk_x, mp_x, ml_x], [mk_y, mp_y, ml_y], [1, 0, ml_z], c=[0, 0, 0, 1])
ax.set_box_aspect([4, 4, 6])
ax.axis('off')
plt.show()