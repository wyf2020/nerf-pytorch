import numpy as np
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int): # 即r为factor
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r) # eg: magick mogrify -resize 50%
        else: # r为resolution
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0]) # eg: magick mogrify -resize 256*256
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1] # 用于获得文件后缀名
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)]) #ext为原格式,png为目标格式
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png': # mogrify 如果原格式和目标格式不同,则不会修改原文件,而是创建新文件;否则会修改原文件
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    '''poses_bounds.npy[N,3*5+2],其中最后一维前4列是相机位姿矩阵; 第5列是H,W,f; 6、7列是根据pts3d估计出来的边界'''
    '''
    npy和npz格式的文件为numpy的数据,其中npy为单个array,npz为一个字典
    e.g. np.save('文件名.npy', np.array([1,2])) 
         a = np.load('文件名.npy')
    '''
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # poses[3,5,N]
    bds = poses_arr[:, -2:].transpose([1,0]) # bds[2,N]
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape # sh=[height, weight]
    
    sfx = ''
    '''三种minify图像的方法:设定缩小因子,设定等比例放缩的目标高或宽'''
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir): # minify失败
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles): # 相机标注数据数量与实际图片个数不同
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    '''这里poses[:2, 4, :].shape=(2,N); np.array(sh[:2]).reshape([2, 1]).shape = (2,1), 所以赋值时发生broadcast'''
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # 修改H,W
    poses[2, 4, :] = poses[2, 4, :] * 1./factor # f = f / factor
    # 如果这里的图像处理是截取中心H,W部分,那么f不用乘(1/factor), 但是这里的图像处理是下采样,如果f不乘(1/factor) 那么投影变换的逆变换 x/z != w/(2*f)
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True) # 读入png文件的gamma correction信息
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles] # 需要把rgb映射到[0,1]
    imgs = np.stack(imgs, -1)  # imgs[H,W,3,N]
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2)) # 叉乘得到vec0为x轴
    vec1 = normalize(np.cross(vec2, vec0)) # 再得到y轴 因为y和z都是平均得到的, 如果直接采用平均的y, 可能会y轴和z轴不垂直
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0) # xyz求平均
    vec2 = normalize(poses[:, :3, 2].sum(0)) # 各坐标系z轴在世界坐标系中的坐标求平均
    up = poses[:, :3, 1].sum(0) # y轴求和
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w


'''生成螺旋线轨迹'''
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) # xyz方向各自的振幅分别取 大于90%相机原点的值
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.]))) # 这里focal为之前根据3d点的上下界估计出来的物体中心的z值, 所以这里相机的朝向是从螺旋线上的各个点面向该中心
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0 # 用于不改变hwf
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses # w2c*poses 把所有的poses从c2w变成c2c, 即从各图像的相机坐标系变换成平均相机坐标系(新的世界坐标系)
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1) # 从[N,3,4]添加[0,0,0,1]变成[N,4,4]
    
    rays_d = poses[:,:3,2:3] # 选取各相机z轴
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d): # 返回距离N个z轴直线最短的点作为新世界坐标系的原点
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4]) # 将世界坐标系变为c2w(平均相机坐标系)

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1))) # 到中心平均距离作为半径
    
    '''令半径平均为1'''
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0) # 求各相机坐标中心
    zh = centroid[2] # 这里zh不一定为0
    radcircle = np.sqrt(rad**2-zh**2) # 各相机柱面坐标系下的半径
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh]) # 渲染相机位置坐标取柱面上等半径一圈120个点
        up = np.array([0,0,-1.]) # 所有y轴取[0,0,-1]

        vec2 = normalize(camorigin) # 相机朝着原点,且z为backward,所以z轴在世界坐标系即为相机的位置坐标
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0) # 从list变为nparray
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1) # 在待渲染的poses矩阵上添加hwf
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1) # 修正后的pose上添加hwf
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # 相机坐标系变换到openGL格式下,由[x down,y right,z backward]变换为[x right, y up, z backward] 
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # poses[N,3,5]
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32) #imgs[N,H,W,3]
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32) # bds[N,2]
    
    # Rescale if bd_factor is provided
    '''1/bds.min() 使near为1; bd_factor保证near比1还大一些,使得后续near取1时一定是下界'''
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5. # 所有3d点z值保守的上下界
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz # 根据3d点的z值 非常粗糙地估计即将生成的螺旋线轨迹的焦距

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0) # rads为[3], 在xyz值上分别大于90%相机原点
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists) # 便于后续auto llff holdout选取测试集时多个图像到中心距离分布均匀
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test



