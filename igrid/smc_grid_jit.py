import numpy as np
from numba import njit

@njit # 连续性约束
def dilation(img, layer):
    msk = img >= layer
    buf = np.copy(msk)
    h, w = img.shape
    for r in range(1, h-1):
        for c in range(1, w-1):
            buf[r,c] &= msk[r-1,c]
            buf[r,c] &= msk[r+1,c]
            buf[r,c] &= msk[r,c-1]
            buf[r,c] &= msk[r,c+1]
    for r in range(h):
        buf[r,0] &= msk[r,1]
        buf[r,-1] &= msk[r,-2]
    for c in range(w):
        buf[0,c] &= msk[1,c]
        buf[-1,c] &= msk[-2,c]
    for r in range(1, h-1):
        for c in range(1, w-1):
            if buf[r,c]!=msk[r,c]:
                img[r,c] = layer
                
@njit # 向上池化
def pool(data, layer):
    h, w = data.shape
    buf = np.zeros((h//2, w//2), data.dtype)
    buf[:] = 10
    for r in range(h//2):
        for c in range(w//2):
            r2 = r*2; c2 = c*2;
            buf[r,c] = min(
                data[r2,c2], data[r2+1,c2],
                data[r2,c2+1], data[r2+1,c2+1])
    return buf

@njit # 标记上下级 0:上级, 1:当前, 2:下级
def mark(up, down, layer, n):
    h, w = down.shape
    hu = not up is None
    s, color = 0, 0
    
    for r in range(h):
        for c in range(w):
            if hu and up[r//2,c//2]>=1024:
                down[r,c]=up[r//2,c//2]
            elif down[r,c]>=layer:
                down[r,c]=1; s+=1
            else: down[r,c]=0
    locs = np.zeros((s, 3), np.uint16)
    for r in range(h):
        for c in range(w):
            if down[r,c]==1:
                down[r,c] = n+color
                locs[color] = r, c, layer
                color += 1
    return locs, n + s

@njit # 根据行列号查找关系
def rela(img, rc):
    (h, w), s = img.shape, 0
    remat = np.zeros((len(rc)*4, 2), dtype=np.uint32)
    for i in range(len(rc)):
        r, c, l = rc[i]
        v = img[r,c]
        if r>0 and img[r-1,c]>5 and v>img[r-1,c]:
            remat[s] = v, img[r-1,c]; s +=1
        if r<h-1 and img[r+1,c]>5 and v>img[r+1,c]:
            remat[s] = v, img[r+1,c]; s +=1
        if c>0 and img[r,c-1]>5 and v>img[r,c-1]:
            remat[s] = v, img[r,c-1]; s +=1
        if c<w-1 and img[r,c+1]>5 and v>img[r,c+1]:
            remat[s] = v, img[r,c+1]; s +=1
    return remat[:s]
    
# 网格剖分函数
def build(data, layer=[1,2,4,8], ct=True):
    im, n, locs, rels = [data, ], 1024, [], []
    if ct: dilation(im[-1], 1)
    for i in layer[1:]:
        im.append(pool(im[-1], i))
        if ct: dilation(im[-1], i)
    index = zip([None]+im[:0:-1], im[::-1], layer[::-1])
    for up, down, l in index:
        rc, n = mark(up, down, l, n); locs.append(rc)
    rels = [rela(i, rc) for i, rc in zip(im[::-1], locs)]
    return data, np.vstack(locs), np.vstack(rels)

if __name__ == '__main__':
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    from time import time
    # 数据读取
    data = Dataset('../SMCGRID1500m.nc')
    
    img = (data.variables['Band1'][::-1]<=0)*8
    
    start = time()
    grid, locs, rels = build(img.copy())
    print(time()-start)

    start = time()
    grid, locs, rels = build(img)
    print(time()-start)

    # 绘图展示，这里用随机颜色
    lut = np.random.randint(3,10, len(locs)+1024, dtype=np.uint8)
    lut[0] = 0
    plt.imshow(lut[grid], cmap='gray')
    plt.show()
    
    
    
