print("all合并")

#乘以指数再合并
import torch
from feature_extractors.features import Features
from utils.mvtec3d_util import *
import numpy as np
import math
import os
from models.feature_fusion import FeatureFusionBlock
import open3d as o3d
import torch.nn.functional as F
FUSION_BLOCK= True
from scipy.io import savemat
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def start_xvfb():
    global xvfb_pid
    xvfb_pid = os.spawnlp(os.P_NOWAIT, "Xvfb", "Xvfb",
                         ":1", "-screen", "0", "1920x1080x24",
                         "-nolisten", "tcp")
    os.environ['DISPLAY'] = ':1'
    atexit.register(stop_xvfb)

def stop_xvfb():
    if xvfb_pid is not None:
        os.kill(xvfb_pid, signal.SIGTERM)

# 或者使用虚拟帧缓冲
# os.environ['PYOPENGL_PLATFORM'] = 'egl'  # 替代方案
class DoubleRGBFPFHFeatures_add_late(Features):
    def calculate_entropy(self,output):
        probabilities = F.softmax(output, dim=0)
        #log_probabilities = torch.log(probabilities)
        log_probabilities = F.log_softmax(output, dim=0)

        entropy = -torch.sum(probabilities * log_probabilities)
        return entropy
    def calculate_gating_weights(self, encoder_output_1, encoder_output_2):
        entropy_1 = np.log(self.calculate_entropy(encoder_output_1))
        entropy_2 = np.log(self.calculate_entropy(encoder_output_2))

        sum_weights = entropy_1 + entropy_2

        entropy_1 /= sum_weights
        entropy_2 /= sum_weights

        return entropy_1, entropy_2
    def l2_normalize(self,features):
        norms = np.linalg.norm(features, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return features / norms

    # DAfpfh
    def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):

        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # [H,W,3] -> [N,3]
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_np)  # 自定义函数

        nonzero_mask = np.all(unorganized_pc != 0, axis=1)
        nonzero_indices = np.nonzero(nonzero_mask)[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        bbox = o3d_pc.get_axis_aligned_bounding_box()
        avg_size = np.mean(bbox.get_extent())
        voxel_scales = [0.05]
        all_fpfh_features = []
        for scale in voxel_scales:
            current_voxel_size = avg_size * scale
            radius_feature = current_voxel_size * 5  # 特征计算半径

            # 1. 计算法线
            o3d_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=current_voxel_size, max_nn=30
                )
            )
            # 2. 计算FPFH特征 (33维)
            pcd_fpfh = o3d.registration.compute_fpfh_feature(
                o3d_pc,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius_feature, max_nn=100
                )
            )
            fpfh = pcd_fpfh.data.T  # [N, 33]

            all_fpfh_features.append(fpfh)
        """计算每个点的局部密度特征"""  # 3个特征
        points = np.asarray(o3d_pc.points)
        kdtree = o3d.geometry.KDTreeFlann(o3d_pc)
        densities = np.zeros(len(points))
        densities2 = np.zeros(len(points))
        densities3 = np.zeros(len(points))
        densities4 = np.zeros(len(points))
        # densities5 = np.zeros(len(points))
        for i in range(len(points)):
            [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.03)
            densities[i] = k  # 邻域内点的数量作为密度
            [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.07)
            densities2[i] = k2  # 邻域内点的数量作为密度
            [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.11)
            densities3[i] = k3  # 邻域内点的数量作为密度
            [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
            densities4[i] = k4
            #     [k5, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
        #     densities4[i] = k5  # 邻域内点的数量作为密度

        fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 4))

        fpfh_with_density[:, :-4] = fpfh
        #fpfh_with_density[:, -5] = densities5
        fpfh_with_density[:, -4] = densities4
        fpfh_with_density[:, -3] = densities3
        fpfh_with_density[:, -2] = densities2
        fpfh_with_density[:, -1] = densities  # 最后一维是密度特征
        full_fpfh_with_density = np.zeros((unorganized_pc.shape[0], fpfh_with_density.shape[1]))
        full_fpfh_with_density[nonzero_indices, :] = fpfh_with_density  # 填充非零点
        full_fpfhwithden_reshaped = full_fpfh_with_density.reshape(
            (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[
                1] + 4))  # full_fpfh_reshaped 使用 reshape 将 full_fpfh 重塑为一个三维数组，形状为 (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1])。这意味着最终的数组将对应于一个有序点云，每个点都有其相应的特征。
        full_fpfhwithden_tensor = torch.tensor(full_fpfhwithden_reshaped).permute(2, 0, 1).unsqueeze(dim=0)

        return full_fpfhwithden_tensor














    # # 改进的fpfh，只有自适应大小没有多尺度密度
    # def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
    #     """
    #     计算有序点云的FPFH特征并拼接三种不同体素大小的特征
    #
    #     参数:
    #         organized_pc: torch.Tensor 形状为 [1, 3, H, W] 的有序点云张量
    #         voxel_size: float 用于计算特征半径的基础体素大小
    #         savefpfh_path: str 特征保存路径(当前未使用但保留接口)
    #
    #     返回:
    #         torch.Tensor 形状为 [1, C, H, W] 的特征张量，其中C=3×FPFH维度(99)
    #     """
    #     # 输入检查和转换
    #     print("输入点云类型:", type(organized_pc))
    #     print("输入点云形状:", organized_pc.shape)
    #
    #     # 转换为无序点云格式 [N, 3]
    #     organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # [H,W,3] -> [N,3]
    #     unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_np)  # 自定义函数
    #
    #     # 移除全零点(无效点)
    #     nonzero_mask = np.all(unorganized_pc != 0, axis=1)
    #     nonzero_indices = np.nonzero(nonzero_mask)[0]
    #     unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    #
    #     # 创建Open3D点云对象
    #     o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    #     bbox = o3d_pc.get_axis_aligned_bounding_box()
    #     avg_size = np.mean(bbox.get_extent())
    #     voxel_scales = [0.05]
    #     all_fpfh_features = []
    #     for scale in voxel_scales:
    #         current_voxel_size = avg_size * scale
    #         radius_feature = current_voxel_size * 5  # 特征计算半径

    #         # 1. 计算法线
    #         o3d_pc.estimate_normals(
    #             search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #                 radius=current_voxel_size, max_nn=30
    #             )
    #         )
    #         # 2. 计算FPFH特征 (33维)
    #         pcd_fpfh = o3d.registration.compute_fpfh_feature(
    #             o3d_pc,
    #             o3d.geometry.KDTreeSearchParamHybrid(
    #                 radius=radius_feature, max_nn=100
    #             )
    #         )
    #         fpfh = pcd_fpfh.data.T  # [N, 33]

    #         all_fpfh_features.append(fpfh)
    #     # """计算每个点的局部密度特征"""  # 3个特征
    #     # points = np.asarray(o3d_pc.points)
    #     # kdtree = o3d.geometry.KDTreeFlann(o3d_pc)
    #     # densities = np.zeros(len(points))
    #     # densities2 = np.zeros(len(points))
    #     # densities3 = np.zeros(len(points))
    #     # densities4 = np.zeros(len(points))
    #     # # densities5 = np.zeros(len(points))
    #     # for i in range(len(points)):
    #     #     [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.05)
    #     #     densities[i] = k  # 邻域内点的数量作为密度
    #     #     [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.1)
    #     #     densities2[i] = k2  # 邻域内点的数量作为密度
    #     #     [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
    #     #     densities3[i] = k3  # 邻域内点的数量作为密度
    #     #     [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.2)
    #     #     densities4[i] = k4  # 邻域内点的数量作为密度

    #     # fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 4))

    #     #
    #     # fpfh_with_density[:, :-4] = fpfh
    #     # fpfh_with_density[:, -4] = densities4
    #     # fpfh_with_density[:, -3] = densities3
    #     # fpfh_with_density[:, -2] = densities2
    #     # fpfh_with_density[:, -1] = densities  # 最后一维是密度特征
    #     full_fpfh_with_density = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]))
    #     full_fpfh_with_density[nonzero_indices, :] = fpfh  # 填充非零点
    #     full_fpfhwithden_reshaped = full_fpfh_with_density.reshape(
    #         (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[
    #             1]))  # full_fpfh_reshaped 使用 reshape 将 full_fpfh 重塑为一个三维数组，形状为 (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1])。这意味着最终的数组将对应于一个有序点云，每个点都有其相应的特征。
    #     full_fpfhwithden_tensor = torch.tensor(full_fpfhwithden_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
    #     print("full_fpfhwithden_tensor ", full_fpfhwithden_tensor.shape)
    #     return full_fpfhwithden_tensor
    #





    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)

        ############### FPFH PATCH ###############
        print("type(sample[1]",type(sample[1]))#type(sample[1] <class 'torch.Tensor'>
        print("sample[1].shape",sample[1].shape)#([1, 3, 224, 224])
        fpfh_feature_maps = self.get_fpfh_features(sample[1])#sample[1]torch.float32非常稀疏   fpfh_feature_mapstorch.float64,33,224,224
        print("sample[1].shape", sample[1].shape)
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))#1,33,28,28 torch.float64
        fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T#torch.float64784 33
        ############### END FPFH PATCH ###############
        print("fpfh_patch.shape, rgb_patch_resize.shape",fpfh_patch.shape, rgb_patch_resize.shape)
        patch = torch.cat([fpfh_patch, rgb_patch_resize], dim=1)
        print("patch.shape", patch.shape)
        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1
        self.patch_xyz_lib.append(fpfh_patch)
        self.patch_rgb_lib.append(rgb_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T


        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(fpfh_patch, rgb_patch, depth_feature_maps_resized[0].shape[-2:], mask, label, center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):


        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())


        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        # 2D dist
        xyz_patch = (fpfh_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        print("xyz_patch.shape,self.patch_xyz_lib.shape", xyz_patch.shape,self.patch_xyz_lib.shape)
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s = torch.tensor([[s_xyz, s_rgb]])
        s_map = torch.cat([s_map_xyz, s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)
        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean)/self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean)/self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

        # print("下采样后self.patch_xyz_lib.shape",self.patch_xyz_lib.shape)
        # print("下采样后self.patch_rgb_lib.shape",self.patch_rgb_lib.shape)
        # print("下采样后self.patch_fusion_lib.shape",self.patch_fusion_lib.shape)


    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''
        # 2D dist
        xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        print("type(self.patch_xyz_lib)",type(self.patch_xyz_lib))
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        xyz_conf,rgb_conf = self.calculate_gating_weights(xyz_patch, rgb_patch)

        s = (xyz_conf * s_xyz + s_rgb* rgb_conf)

        s_map = (xyz_conf *s_map_xyz + s_map_rgb* rgb_conf)
        s_map = s_map.view(1, self.image_size, self.image_size)
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)/1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal=='xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal=='xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        s_map = self.blur(s_map)
        return s, s_map




