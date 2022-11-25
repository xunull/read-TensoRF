from .tensorBase import *


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        """
        aabb: 框
        gridSize: 分辨率
        device:
        """
        super(TensorCP, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):
        # init 时候会被调用
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)

        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        """
        n_component: Rsigma或者是Rc，或者是CP中的同一个值
        gridSize: 分辨率
        scale: 0.2
        """

        line_coef = []
        # vecMode [2,1,0]
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]

            # 96个组件，每个组件的向量，长度是128
            # like [1,96,128,1]
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))

        return torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr': lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):
        # 公式9
        # 取出xyz中的各个分量，顺序是210，然后堆叠在一起，从[bs*N_mask,3] -> [3,bs*N_mask]
        # xyz_sampled (bs,3) vecMode [2, 1, 0]
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]],
             xyz_sampled[..., self.vecMode[1]],
             xyz_sampled[..., self.vecMode[2]]))
        # 在坐标分量前面堆叠一个0 [3,bs*N_mask] -> [3,bs*N_mask,2] -> view -> [3,bs*N_mask,1,2]
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line),
                                       coordinate_line), dim=-1).detach().view(3, -1, 1, 2)  # (3,bs,1,2)
        # 因为在(128,1)的hw的xy中x只有一项，因此上面的堆叠的第一个值全是0，第二个值会在128个值上线性插值
        # (Rsigma,bs*N_mask)
        # coordinate_line[[0]] (1,bs*N_mask,1,2)
        # density_line[0] Parameter [1,96,128,1] 96是Rsigma,128是分辨率,96类似于通道数,(128,1)类似于HW，因为W只有1个，所有sn
        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])

        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        # 将Rsigma上的值加和，得到所有点的sigma (Rsigma,bs*N_mask) -> (bs*N_mask)
        sigma_feature = torch.sum(line_coef_point, dim=0)  # (16,bs) -> (bs)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled):
        # 这里的点位可能比计算sigma那里的少，因为会经过一个阈值的筛选
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]],
             xyz_sampled[..., self.vecMode[1]],
             xyz_sampled[..., self.vecMode[2]]))

        coordinate_line = torch.stack((torch.zeros_like(coordinate_line),
                                       coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(self.app_line[0], coordinate_line[[0]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])

        line_coef_point = line_coef_point * F.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        # (288,bs)
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear',
                              align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0]:b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0]:b_r[mode0], :]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        # 计算loss的一个方法，会在train中被调用
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        # 计算loss的一个方法，会在train中被调用
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        # 计算loss的一个方法，会在train中被调用
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total
