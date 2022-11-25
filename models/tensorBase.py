import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time


# 位置编码
def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # 体渲染
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb


# ----------------------------------------------------------------------------------------------------------------------

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(
            self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


# ----------------------------------------------------------------------------------------------------------------------

class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()
        # input channel的大小
        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            # 位置编码
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            # 视角编码
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


# ----------------------------------------------------------------------------------------------------------------------

class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


# ----------------------------------------------------------------------------------------------------------------------

class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


# ----------------------------------------------------------------------------------------------------------------------

class TensorBase(torch.nn.Module):
    def __init__(self,
                 # 框
                 aabb,
                 # 分辨率
                 gridSize,
                 # cuda or cpu
                 device,
                 # R sigma
                 density_n_comp=8,
                 # Rc
                 appearance_n_comp=24,
                 app_dim=27,
                 shadingMode='MLP_PE',
                 alphaMask=None,
                 near_far=[2.0, 6.0],
                 density_shift=-10,
                 alphaMask_thres=0.001,
                 # todo
                 distance_scale=25,
                 rayMarch_weight_thres=0.0001,
                 pos_pe=6, view_pe=6, fea_pe=6, featureC=128, step_ratio=2.0,
                 fea2denseAct='softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio  # todo

        # 更新分辨率
        self.update_stepSize(gridSize)

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        # todo 这个顺序
        self.vecMode = [2, 1, 0]
        # todo
        self.comp_w = [1, 1, 1]

        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe = shadingMode, pos_pe, view_pe, fea_pe
        self.featureC = featureC

        # 初始化渲染函数，MLP，带PE的MLP，球协函数等
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            # 使用位置编码的MLP
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()

        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    # 在__init__中会先被调用一次
    # 分辨率变化的时候也会被调用
    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        # like [3,3,3]
        self.aabbSize = self.aabb[1] - self.aabb[0]
        # like [2/3,2/3,2/3]
        self.invaabbSize = 2.0 / self.aabbSize
        # like [128,128,128]
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        # 一个体素格子的大小
        self.units = self.aabbSize / (self.gridSize - 1)

        self.stepSize = torch.mean(self.units) * self.step_ratio
        # 立体对角线的长度
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        # 立体对角线的长度/步长 = 采样点的数量
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1

        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1  # todo invaabbSize is 2/3 最后为什么要减1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def shrink(self, new_aabb, voxel_size):
        pass

    def get_kwargs(self):
        # 模型的各种参数
        return {
            'aabb': self.aabb,
            'gridSize': self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        # 保存模型
        # 将很多参数保存
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        # 加载模型
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device),
                                           alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    # 在光线上进行采样
    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):

        N_samples = N_samples if N_samples > 0 else self.nSamples

        stepsize = self.stepSize

        near, far = self.near_far

        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)

        rate_a = (self.aabb[1] - rays_o) / vec

        rate_b = (self.aabb[0] - rays_o) / vec
        # torch.minimum(rate_a, rate_b) -> [bs,3] , amax(-1) -> (bs,)
        # 采样的起始位置
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()

        if is_train:
            # (bs,samples)
            rng = rng.repeat(rays_d.shape[-2], 1)

            rng += torch.rand_like(rng[:, [0]])

        step = stepsize * rng.to(rays_o.device)
        # 插值点，采样点
        interpx = (t_min[..., None] + step)
        # 得到光线上的采样点
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # 标记超出框的点, 后面的~就是在框内的点
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)

        dense_xyz = dense_xyz.transpose(0, 2).contiguous()

        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        # 体素数量
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        # like [1,1,128,128,128] -> [128,128,128]
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])

        alpha[alpha >= self.alphaMask_thres] = 1

        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)
        # alpha的取值也只有0和1
        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)

        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)

        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))

        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False):
        """
        all_rays: 场景中的所有光线
        all_rgbs: 所有图像上的像素
        """
        # 这个方法是在forward之外被调用
        print('========> filtering rays ...')
        tt = time.time()
        # 光线的数量
        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        # 分块
        idx_chunks = torch.split(torch.arange(N), chunk)

        for idx_chunk in idx_chunks:
            # 某批光线
            rays_chunk = all_rays[idx_chunk].to(self.device)
            # rays_d 是单位向量
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]

            if bbox_only:

                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                # max point - origin
                rate_a = (self.aabb[1] - rays_o) / vec
                # min point - origin
                rate_b = (self.aabb[0] - rays_o) / vec
                # min max
                t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
                # max min
                t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)

                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)

                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time() - tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')

        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2density(self, density_features):
        # 激活函数
        if self.fea2denseAct == "softplus":
            # SoftPlus 是 ReLU 函数的平滑近似，可用于约束机器的输出始终为正
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        # nerf中的alpha计算公式
        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        # [bs,6]
        # sample points, [bs,3]
        viewdirs = rays_chunk[:, 3:6]

        if ndc_ray:

            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                 N_samples=N_samples)

            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)

            dists = dists * rays_norm

            viewdirs = viewdirs / rays_norm
        else:
            # 在光线上进行点采样
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                             N_samples=N_samples)

            # 两个点之间的距离（并非空间上的点）
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        # (N,3) -> (bs,N,3)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])

            alpha_mask = alphas > 0

            ray_invalid = ~ray_valid

            ray_invalid[ray_valid] |= (~alpha_mask)

            ray_valid = ~ray_invalid

        # 输出的颜色和体积密度 (bs,n)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        # (bs,n,3)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        # 至少有个点在box内
        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)

            # 调用子类的
            # 计算 体积密度那个G公式
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            # 将值经过激活函数，得到最后的体积密度，就是这里的函数，不是子类的
            validsigma = self.feature2density(sigma_feature)

            # 填充上计算的结果
            sigma[ray_valid] = validsigma

        # 体渲染中的alpha,weight那些
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        # 体积密度要大于一定的值才会计算appearance
        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            # 调用子类的
            # (bs, 27) 27是初始化时候的参数，在这里可能经过的一个chan变换为 288 -> 27
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            # (bs,3)
            # 里面有神经网络，就是这里的函数不是子类的
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)

            # 填充上计算的结果
            rgb[app_mask] = valid_rgbs

        # 累积权重值
        acc_map = torch.sum(weight, -1)

        # 体渲染 weights * rgb
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        # 限制在0-1
        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():

            depth_map = torch.sum(weight * z_vals, -1)

            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        # 只用了两个返回值，其他的类似于NeRF中的返回值在这个实验中没用
        return rgb_map, depth_map  # rgb, sigma, alpha, weight, bg_weight
