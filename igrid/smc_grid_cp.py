import cupy as np

# 准备四叉树索引
s1 = slice(0,None,2), slice(0,None,2)
s2 = slice(0,None,2), slice(1,None,2)
s3 = slice(1,None,2), slice(0,None,2)
s4 = slice(1,None,2), slice(1,None,2)
ss = s1, s2, s3, s4

# 保证连续
def dilation(img, layer):
    msk = img>=layer
    buf = msk.copy()
    buf[1:] &= msk[:-1]
    buf[:-1] &= msk[1:]
    buf[:,1:] &= msk[:,:-1]
    buf[:,:-1] &= msk[:,1:]
    img *= buf
    layer = np.array(layer, img.dtype)
    msk &= ~buf; 
    img += msk * layer
    
# 向上池化
def pool(data, layer):
    buf = data[ss[0]].copy()
    for i in ss[1:]:
        np.minimum(data[i], buf, out=buf)
    return buf

# 标记上下级 0:上级, 1:当前, 2:下级
def mark(up, down, layer, n):
    np.greater_equal(down, layer, out=down)
    if not up is None:
        for i in ss: np.maximum(up, down[i], down[i])
    r, c = np.where(down ==1)
    down[r, c] = np.arange(n, n+len(r))
    v = np.ones(len(r), dtype=np.uint16)*layer
    rcv = np.hstack((r, c, v)).reshape(3,-1).T
    return rcv, n+len(r)

# 根据行列号查找关系
def rela(img, rc):
    if len(rc)==0: return np.zeros((0,2), img.dtype)
    (h, w), (r, c, l) = img.shape, rc.T
    v = img[r, c]
    tp = v, img[np.clip(r, 1, None)-1, c]
    bt = v, img[np.clip(r, None, h-2)+1, c]
    lt = v, img[r, np.clip(c, 1, None)-1]
    rt = v, img[r, np.clip(c, None, w-2)+1]
    remat = np.hstack([tp, bt, lt, rt]).T
    msk = remat[:,0] > remat[:,1]
    msk &= remat[:,1] != 0
    return remat[msk]

# 网格剖分函数
def build(data, layer=[1,2,4,8], ct=True):
    im, n, locs, rels = [data, ], 1024, [], []
    if ct: dilation(img, 1)
    for i in layer[1:]:
        im.append(pool(im[-1], i))
        if ct: dilation(im[-1], i)
    index = zip([None]+im[:0:-1], im[::-1], layer[::-1])
    for up, down, l in index:
        rc, n = mark(up, down, l, n); locs.append(rc)
    rels = [rela(i, rc) for i, rc in zip(im[::-1], locs)]
    return data, np.vstack(locs), np.vstack(rels)

if __name__ == '__main__':
    from imageio import imread
    import matplotlib.pyplot as plt
    from time import time
    # 数据读取
    img = imread('distance.tif').astype(np.int32)
    img = np.array(img)
    '''
    dis = ndimg.distance_transform_cdt(img>0)
    lut = np.array([0,1]+[2]*2+[4]*4+[8]*8+[16]*2048)
    img = (np.array(lut)[dis]).astype(np.int32)
    '''
    img = (img>0)*8

    start = time()
    grid, locs, rels = build(img.copy())
    print(time()-start)

    start = time()
    grid, locs, rels = build(img)
    print(time()-start)
    '''
    grid, locs, rels = [np.asnumpy(i) for i in (grid, locs, rels)]

    # 绘图展示，这里用随机颜色
    lut = np.random.randint(3,10, len(locs)+1024, dtype=np.uint8)
    lut[0] = 0; lut = np.asnumpy(lut)
    plt.imshow(lut[grid], cmap='gray')
    plt.show()
    '''
    
    '''
    # 以下是一个小型测试数据，带有关系图绘制
    data = np.array([[1,1,2,2,4,4,4,4],
                     [1,2,2,2,4,4,4,4],
                     [1,1,1,2,4,4,4,4],
                     [1,2,2,4,4,4,4,4],
                     [2,2,4,4,4,4,4,4],
                     [2,2,4,4,4,4,4,4],
                     [2,2,4,4,4,4,4,2],
                     [1,2,2,4,4,4,2,2]], dtype=np.uint32)

    grid, locs, rels = build(data.copy())
    #grid, locs, rels = [np.asnumpy(i) for i in (grid, locs, rels)]

    # 绘制示意图
    ax1, ax2 = plt.subplot(121), plt.subplot(122)
    ax1.imshow(data)
    ax2.imshow(grid)
    rs, cs = (locs[:,:2]+0.5).T*locs[:,2]-0.5
    ax2.plot(cs, rs, 'r.')
    for r, c in zip(rs, cs):
        ax2.text(c, r, str(grid[int(r), int(c)]), color='red')
    for s, e in rels-1024:
        plt.plot(cs[[s,e]], rs[[s,e]], 'b-')
    plt.show()
    '''
