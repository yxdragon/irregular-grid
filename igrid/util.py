import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit

@njit
def render(rel):
    rel = rel.copy()
    cs = np.zeros(rel.max()+1, dtype=np.uint8)
    cs[rel.ravel()] = 1
    for i in range(15):
        np.random.shuffle(rel)
        s = 0
        for i in range(len(rel)):
            if cs[rel[i,0]]==cs[rel[i,1]]:
                cs[rel[i,0]] = cs[rel[i,0]]%8+1
                s += 1
        if s==0: break
    return cs

def render(grid, locs, rels):
    rels = rels.T.copy()
    cs = np.zeros(rels.max()+1, dtype=np.uint8)
    rd = np.random.randint(1, 9, len(cs))
    cs[rels.ravel()] = rd[rels.ravel()]
    for i in range(15):
        np.random.shuffle(rels)
        i1, i2 = rels
        same = cs[i1] == cs[i2]
        ii = i1[same][::2]
        cs[ii] %= 8; cs[ii]+=1
        if same.sum()==0: return cs
    return cs[grid]

def plot_grid(img, locs, rels, rel=True):
    plt.imshow(img)
    if rel:
        rs, cs = (locs[:,:2]+0.5).T*locs[:,2]-0.5
        line = np.hstack((rels, rels[:,:1]))-1024
        plt.triplot(cs, rs, line, color='red')
    plt.show()
    
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    x, y = np.ogrid[-2:2:160j, -2:2:160j]
    z = abs(x) * np.exp(-x ** 2 - (y / .75) ** 2)
    z = (20 - z/z.max()*20).astype(np.int32)

    from multi_scale_grid_np import build
    z[z>0] = 16
    z[:4] = 1
    grid, locs, rels = build(z, [1,2,4,8,16], True)

    grid = render(grid, locs, rels)
    plot_grid(grid, locs, rels, False)
