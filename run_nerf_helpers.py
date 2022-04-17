import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2) # 计算两个图像的mse(均方误差)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])) # 这里的x一般为img2mse(x,y),该函数进一步计算两个图像的PSNR
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8) # 把rgb从[0,1]浮点变成[0,255]整数


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']: # 是否包括输入的坐标
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        '''如果log_sampling则在[1,2^(multires-1)]对数采样,反之则线性采样'''
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        '''
        生成一个embed_fns这样一个list,里面有2*multires或2*multires+1个(2来自sin,cos)lambda函数,用来position coding
        out_dim = input_dims*2*multires或input_dims*(2*multires+1) 取决于position coding后是否包括输入的坐标
        '''
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs): # 生成依次用embed_fns的函数生成position coding后的坐标,并在最后一维拼接起来
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0): 
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1, # position coding频率范围为[1,2^(multires-1)]
                'num_freqs' : multires,
                'log_sampling' : True, 
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x) # 把成员函数变成公用的lambda函数
    return embed, embedder_obj.out_dim # 返回embed函数和输出维数


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        '''
        nn.linear是pytorch的一个仿射变换层
        论文中带viewdir的MLP一共有11层,10个隐含层.
        '''
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        # 这行初始化前8个层,其中第6个变换的输入并需要并上输入
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        # 这是第10层,256并上3或24个view,输出128
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W) # 第9层的一部分
            self.alpha_linear = nn.Linear(W, 1) # 第9层的另一部分,生成alpha
            self.rgb_linear = nn.Linear(W//2, 3) # 第11层 输出rgb
        else:
            self.output_linear = nn.Linear(W, output_ch) # 如果没有viewdir 则只有前面初始化的8层加这一层,输出4个或5个, 为什么会有5个呢? 这里应该只有4个的情况 前面run_nerf.py写错了,在github的issue上查到了已经有人提出过这个问题了.

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips: # 如果是第6层,需要拼接输入
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h) # 第9层无激活函数
            h = torch.cat([feature, input_views], -1) # 第10层输入
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h) # 第10层
                h = F.relu(h)

            rgb = self.rgb_linear(h) # 第11层
            outputs = torch.cat([rgb, alpha], -1) # 输出rgba
        else:
            outputs = self.output_linear(h) # 如果没有viewdir,第8层后直接再加一层,输出rgba

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w): # torch版本的一个相机中每一像素的坐标和方向到世界坐标系
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    '''上面三行应该等价于indexing = 'xy'的meshgrid'''
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    '''运用broadcast [W,H,1,3]*[3,3] 会broadcast成[W,H,3,3]*[W,H,3,3],再sum最后一维,得到rays_d为[W,H,3]'''
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape) # [3,1]->[W,H,3]
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w): # numpy版本的一个相机中每一像素的坐标和方向到世界坐标系
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans # weight的格式为[N_rays, N_samples-2] "-2"是因为该函数输入的参数是weights[...,1:-1]
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins)) # 即[N_rays, N_samples]

    # Take uniform samples
    if det: # 如果z_vals没有噪音
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # 将u从[N_samples]变为[N_rays, N_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]) # 得到[N_rays, N_samples]个[0,1]随机数

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest: # u的生成改用随机数种子0, 使结果具有一致性
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous() # 改变数组在内存中的顺序,防止 RuntimeError: input is not contiguous
    inds = torch.searchsorted(cdf, u, right=True) # 对每个u[i]返回一个cdf数组的编号f(i),使得cdf[f(i)-1] <= u[i] < cdf[f(i)]
    below = torch.max(torch.zeros_like(inds-1), inds-1) # below为inds-1,并且限制其大于等于0
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) # above为inds,并且限制其小于等于N-1
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2) # 拼接below,above

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # matched_shape = (N_rays, N_samples, N_samples)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) # cdf_g格式为[N_rays, N_samples, 2],其中cdf_g[i][j][k] = cdf[i][j][inds_g[i][j][k]]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) # bins_g格式同上
    
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
    ''' + t * (bins_g[...,1]-bins_g[...,0])使密集的部分也均匀采样,避免直接使用同样的bins_g[...,0]'''
    return samples
