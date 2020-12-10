import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from util import plot_grid, render

x, y = np.ogrid[-2:2:160j, -2:2:160j]
z = abs(x) * np.exp(-x ** 2 - (y / .75) ** 2)
z = (20 - z/z.max()*20).astype(np.int32)


from smc_grid_np import build

grid, locs, rels = build(z, [1,2,4,8,16], True)
grid = render(rels)[grid]
plot_grid(grid, locs, rels, False)
