from feature_extractors.features import Features
from utils.mvtec3d_util import *
import numpy as np
import math
import os
import open3d as o3d
from DenseSIFTDescriptor import  DenseSIFTDescriptor
from skimage.feature import hog
from feature_extractors.adsfeatures import ADsFeatures
FUSION_BLOCK= True
from models.feature_fusion import FeatureFusionBlock


class DoubleRGBFPFHFeatures_add(Features):
    #原PRletters的
    # def get_fpfh_features(self,organized_pc, voxel_size=0.05):
    #     # # Convert organized PointCloud to numpy array
    #     print("organized_pc.type", type(organized_pc))# <class 'torch.Tensor'>
    #     print("organized_pc.shape", organized_pc.shape)#torch.Size([1, 3, 224, 224])
    #     organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    #     ## Convert organized PointCloud to unorganized PointCloud
    #     unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    #     # # Find nonzero indices and remove zero elements
    #     nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    #     unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    #     # Create Open3D PointCloud object
    #     o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    #     # Estimate normals for the unorganized PointCloud
    #     radius_normal = voxel_size * 2
    #     o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    #     ## Compute Fast Point Feature Histogram (FPFH) features
    #     radius_feature = voxel_size * 5
    #     pcd_fpfh = o3d.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
    #     (radius=radius_feature, max_nn=100))
    #     fpfh = pcd_fpfh.data.T
    #     # Create full FPFH array and reshape it
    #     full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)
    #     full_fpfh[nonzero_indices, :] = fpfh
    #     print("full_fpfh[nonzero_indices, :] = fpfh,fpfh.shape",fpfh.shape)
    #     full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))
    #     # # Convert FPFH features to a PyTorch tensor
    #     full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
    #     return full_fpfh_tensor



    # 改进的fpfhEye
    def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
        """
        计算有序点云的FPFH特征并拼接三种不同体素大小的特征

        参数:
            organized_pc: torch.Tensor 形状为 [1, 3, H, W] 的有序点云张量
            voxel_size: float 用于计算特征半径的基础体素大小
            savefpfh_path: str 特征保存路径(当前未使用但保留接口)

        返回:
            torch.Tensor 形状为 [1, C, H, W] 的特征张量，其中C=3×FPFH维度(99)
        """
        # 输入检查和转换
        print("输入点云类型:", type(organized_pc))
        print("输入点云形状:", organized_pc.shape)

        # 转换为无序点云格式 [N, 3]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # [H,W,3] -> [N,3]
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_np)  # 自定义函数

        # 移除全零点(无效点)
        nonzero_mask = np.all(unorganized_pc != 0, axis=1)
        nonzero_indices = np.nonzero(nonzero_mask)[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

        # 创建Open3D点云对象
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        bbox = o3d_pc.get_axis_aligned_bounding_box()
        avg_size = np.mean(bbox.get_extent())
        voxel_scales = [0.06]
        all_fpfh_features = []
        for scale in voxel_scales:
            current_voxel_size = avg_size * scale
            radius_feature = current_voxel_size * 5  # 特征计算半径
            print(f"体素比例 {scale}: 体素大小 {current_voxel_size:.3f}, 特征半径: {radius_feature:.3f}")
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
            print(f"体素比例 {scale} 的FPFH特征形状: {fpfh.shape}")
            all_fpfh_features.append(fpfh)
        """计算每个点的局部密度特征"""  # 3个特征
        points = np.asarray(o3d_pc.points)
        kdtree = o3d.geometry.KDTreeFlann(o3d_pc)
        densities = np.zeros(len(points))
        densities2 = np.zeros(len(points))
        densities3 = np.zeros(len(points))
        densities4 = np.zeros(len(points))
        #densities5 = np.zeros(len(points))
        for i in range(len(points)):
            [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.05)
            densities[i] = k  # 邻域内点的数量作为密度
            [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.01)
            densities2[i] = k2  # 邻域内点的数量作为密度
            [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
            densities3[i] = k3  # 邻域内点的数量作为密度
            [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.2)
            densities4[i] = k4
            #print("Eye den ")
            # [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.07)
            # densities[i] = k  # 邻域内点的数量作为密度
            # [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.13)
            # densities2[i] = k2  # 邻域内点的数量作为密度
            # [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.19)
            # densities3[i] = k3  # 邻域内点的数量作为密度
            # [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.25)
            # densities4[i] = k4

            # [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.04)
            # densities[i] = k  # 邻域内点的数量作为密度
            # [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.06)
            # densities2[i] = k2  # 邻域内点的数量作为密度
            # [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.08)
            # densities3[i] = k3  # 邻域内点的数量作为密度
            # [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.1)
            # densities4[i] = k4  #
            # [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.04)
            # densities[i] = k  # 邻域内点的数量作为密度
            # [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.06)
            # densities2[i] = k2  # 邻域内点的数量作为密度
            # [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.08)
            # densities3[i] = k3  # 邻域内点的数量作为密度
            # [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.1)
            # densities4[i] = k4
            # [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.03)
            # densities[i] = k  # 邻域内点的数量作为密度
            # [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.07)
            # densities2[i] = k2  # 邻域内点的数量作为密度
            # [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.11)
            # densities3[i] = k3  # 邻域内点的数量作为密度
            # [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
            # densities4[i] = k4

        print("densities", densities, densities2, densities3, densities4)
        fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 4))
        print("fpfh.shape", fpfh.shape)
        print("fpfh_with_density.shape", fpfh_with_density.shape)
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
        print("full_fpfhwithden_tensor ", full_fpfhwithden_tensor.shape)
        return full_fpfhwithden_tensor


    # # 改进的fpfhmvt
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
    #         print(f"体素比例 {scale}: 体素大小 {current_voxel_size:.3f}, 特征半径: {radius_feature:.3f}")
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
    #         print(f"体素比例 {scale} 的FPFH特征形状: {fpfh.shape}")
    #         all_fpfh_features.append(fpfh)
    #     """计算每个点的局部密度特征"""  # 3个特征
    #     points = np.asarray(o3d_pc.points)
    #     kdtree = o3d.geometry.KDTreeFlann(o3d_pc)
    #     densities = np.zeros(len(points))
    #     densities2 = np.zeros(len(points))
    #     densities3 = np.zeros(len(points))
    #     densities4 = np.zeros(len(points))
    #     # densities5 = np.zeros(len(points))
    #     for i in range(len(points)):
    #         [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.03)
    #         densities[i] = k  # 邻域内点的数量作为密度
    #         [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.07)
    #         densities2[i] = k2  # 邻域内点的数量作为密度
    #         [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.11)
    #         densities3[i] = k3  # 邻域内点的数量作为密度
    #         [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
    #         densities4[i] = k4
    #         #     [k5, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
    #     #     densities4[i] = k5  # 邻域内点的数量作为密度
    #     print("densities", densities, densities2, densities3, densities4)
    #     fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 4))
    #     print("fpfh.shape", fpfh.shape)
    #     print("fpfh_with_density.shape", fpfh_with_density.shape)
    #     fpfh_with_density[:, :-4] = fpfh
    #     #fpfh_with_density[:, -5] = densities5
    #     fpfh_with_density[:, -4] = densities4
    #     fpfh_with_density[:, -3] = densities3
    #     fpfh_with_density[:, -2] = densities2
    #     fpfh_with_density[:, -1] = densities  # 最后一维是密度特征
    #     full_fpfh_with_density = np.zeros((unorganized_pc.shape[0], fpfh_with_density.shape[1]))
    #     full_fpfh_with_density[nonzero_indices, :] = fpfh_with_density  # 填充非零点
    #     full_fpfhwithden_reshaped = full_fpfh_with_density.reshape(
    #         (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[
    #             1] + 4))  # full_fpfh_reshaped 使用 reshape 将 full_fpfh 重塑为一个三维数组，形状为 (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1])。这意味着最终的数组将对应于一个有序点云，每个点都有其相应的特征。
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
        fpfh_feature_maps = self.get_fpfh_features(sample[1])#sample[1]torch.float32
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))#1,33,28,28 torch.float64
        fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T#torch.float64784 33
        ############### END FPFH PATCH ###############
        patch = torch.cat([fpfh_patch, rgb_patch_resize], dim=1)


        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name+ str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(fpfh_patch)
        #print("type(xyz_patch),type(self.patch_lib)",type(xyz_patch),type(self.patch_lib))#<class 'torch.Tensor'> <class 'list'>
        #print("type(patch)",type(patch))#type(patch) <class 'torch.Tensor'>
        self.patch_rgb_lib.append(rgb_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        # xyz_patch = torch.cat(xyz_feature_maps, 1)
        # xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size*self.image_size), dtype=xyz_patch.dtype)
        # xyz_patch_full[:,:,nonzero_indices] = interpolated_pc
        # xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        # xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        # xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

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
        #s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb]])
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

        s = s_xyz + s_rgb
        s_map = s_map_xyz + s_map_rgb
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
        print("s_star = torch.max(min_val)/1000",s_star)

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
class DoubleRGBFPFHFeatures(Features):

    def get_fpfh_features(self,organized_pc, voxel_size=0.05):
        # # Convert organized PointCloud to numpy array
        print("organized_pc.type", type(organized_pc))
        print("organized_pc.shape", organized_pc.shape)
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()#将原来的第一个维度（通常是批次维度）移动到最后一个位置
        ## Convert organized PointCloud to unorganized PointCloud
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        # # Find nonzero indices and remove zero elements
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        # Create Open3D PointCloud object
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        # Estimate normals for the unorganized PointCloud
        radius_normal = voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        ## Compute Fast Point Feature Histogram (FPFH) features
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T
        # Create full FPFH array and reshape it
        full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)
        full_fpfh[nonzero_indices, :] = fpfh
        print("full_fpfh[nonzero_indices, :] = fpfh,fpfh.shape",fpfh.shape)
        full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))
        # # Convert FPFH features to a PyTorch tensor
        full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
        return full_fpfh_tensor

    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())



        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784 * 4, -1)

        ############### FPFH PATCH ###############
        print("type(sample[1]",type(sample[1]))#type(sample[1] <class 'torch.Tensor'>
        print("sample[1].shape",sample[1].shape)#([1, 3, 224, 224])
        fpfh_feature_maps = self.get_fpfh_features(sample[1])#sample[1]torch.float32
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))#1,33,28,28 torch.float64
        fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T#torch.float64784 33
        ############### END FPFH PATCH ###############
        patch = torch.cat([fpfh_patch, rgb_patch_resize], dim=1)


        if class_name is not None:
            torch.save(patch, os.path.join(self.args.save_feature_path, class_name + str(self.ins_id) + '.pt'))
            self.ins_id += 1

        self.patch_xyz_lib.append(fpfh_patch)
        self.patch_rgb_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        # xyz_patch = torch.cat(xyz_feature_maps, 1)
        # xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
        #                              dtype=xyz_patch.dtype)
        # xyz_patch_full[:, :, nonzero_indices] = interpolated_pc
        # xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        # xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        # xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T

        self.compute_s_s_map(fpfh_patch, rgb_patch, depth_feature_maps_resized[0].shape[-2:], mask, label, center,
                             neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_late_fusion_mem_bank(self, sample):

        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())

        # xyz_patch = torch.cat(xyz_feature_maps, 1)
        # xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
        #                              dtype=xyz_patch.dtype)
        # xyz_patch_full[:, :, nonzero_indices] = interpolated_pc
        # xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        # xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        # xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T


        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T

        # 2D dist
        xyz_patch = (fpfh_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb]])

        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(2, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std

        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 2D dist
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb]])
        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb],
                          dim=0).squeeze().reshape(2, -1).permute(1, 0)
        # print("multi456s.shape", s.shape)#torch.Size([1, 2])
        # print("s_map = torch.cat(",s_map.shape)#torch.Size([50176, 2])
        s = torch.tensor(self.detect_fuser.score_samples(s))
        # print("multi458s.shape", s.shape)# torch.Size([1])
        # print("s = torch.tensor(self.detect_fuser.score_samples(s))",s_map.shape)#torch.Size([50176, 2])
        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        # print("s_map = torch.tensor(self.seg_fuser.score_samples(s_map))",s_map.shape)#torch.Size([50176])=224*224
        s_map = s_map.view(1, 224, 224)
        # print(" s_map = s_map.view(1, 224, 224)", s_map.shape)#torch.Size([1, 224, 224])

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val) / 1000

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal == 'xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1

        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        if modal == 'xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1) / 1000
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1) / 1000

        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

class FPFHFeatures(Features):
# #原本的fpfh
#     def get_fpfh_features(self,organized_pc, voxel_size=0.05):
#         # # Convert organized PointCloud to numpy array
#         print("organized_pc.type", type(organized_pc))
#         print("organized_pc.shape", organized_pc.shape)
#         print("FPFH尺寸体素")
#         organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()#将原来的第一个维度（通常是批次维度）移动到最后一个位置
#         ## Convert organized PointCloud to unorganized PointCloud
#         unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
#         # # Find nonzero indices and remove zero elements
#         nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
#         unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
#         # Create Open3D PointCloud object
#         o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
#         # Estimate normals for the unorganized PointCloud
#         radius_normal = voxel_size * 2
#         o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
#         ## Compute Fast Point Feature Histogram (FPFH) features
#         radius_feature = voxel_size * 5
#         pcd_fpfh = o3d.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
#         (radius=radius_feature, max_nn=100))
#         fpfh = pcd_fpfh.data.T
#         # Create full FPFH array and reshape it
#         full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)
#         full_fpfh[nonzero_indices, :] = fpfh
#         print("full_fpfh[nonzero_indices, :] = fpfh,fpfh.shape",fpfh.shape)
#         full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))
#         # # Convert FPFH features to a PyTorch tensor
#         full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
#         return full_fpfh_tensor
#改进的fpfh
    def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
        """
        """
        # 输入检查和转换
        print("输入点云类型:", type(organized_pc))
        print("输入点云形状:", organized_pc.shape)

        # 转换为无序点云格式 [N, 3]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # [H,W,3] -> [N,3]
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_np)  # 自定义函数

        # 移除全零点(无效点)
        nonzero_mask = np.all(unorganized_pc != 0, axis=1)
        nonzero_indices = np.nonzero(nonzero_mask)[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

        # 创建Open3D点云对象
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        bbox = o3d_pc.get_axis_aligned_bounding_box()
        avg_size = np.mean(bbox.get_extent())
        voxel_scales = [0.05]
        all_fpfh_features = []
        for scale in voxel_scales:
            current_voxel_size = avg_size * scale
            radius_feature = current_voxel_size * 5  # 特征计算半径
            print(f"体素比例 {scale}: 体素大小 {current_voxel_size:.3f}, 特征半径: {radius_feature:.3f}")
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
            print(f"体素比例 {scale} 的FPFH特征形状: {fpfh.shape}")
            all_fpfh_features.append(fpfh)

        """densities"""
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
            densities4[i] = k4  # 邻域内点的数量作为密度

        print("densities", densities, densities2, densities3, densities4)
        fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 4))
        print("fpfh.shape", fpfh.shape)
        print("fpfh_with_density.shape", fpfh_with_density.shape)

        fpfh_with_density[:, :-4] = fpfh
        fpfh_with_density[:, -4] = densities4
        fpfh_with_density[:, -3] = densities3
        fpfh_with_density[:, -2] = densities2
        fpfh_with_density[:, -1] = densities
        full_fpfh_with_density = np.zeros((unorganized_pc.shape[0], fpfh_with_density.shape[1]))
        full_fpfh_with_density[nonzero_indices, :] = fpfh_with_density  # 填充非零点
        full_fpfhwithden_reshaped = full_fpfh_with_density.reshape(
            (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[
                1] + 4))
        full_fpfhwithden_tensor = torch.tensor(full_fpfhwithden_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
        print("full_fpfhwithden_tensor ", full_fpfhwithden_tensor.shape)
        return full_fpfhwithden_tensor


    def add_sample_to_mem_bank(self, sample, class_name=None):
        # organized_pc = sample[1]
        # organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        # unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        # nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        #
        # unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        # rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
        #                                                                                              unorganized_pc_no_zeros.contiguous())
        ############### FPFH PATCH ###############
        print("type(sample[1]",type(sample[1]))#type(sample[1] <class 'torch.Tensor'>
        print("sample[1].shape",sample[1].shape)#([1, 3, 224, 224])
        fpfh_feature_maps = self.get_fpfh_features(sample[1])#sample[1]torch.float32
        fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))#1,33,28,28 torch.float64
        fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1], -1).T#torch.float64784 33
        ############### END FPFH PATCH ###############
        self.patch_lib.append(fpfh_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                                     unorganized_pc_no_zeros.contiguous())


        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        self.compute_s_s_map(fpfh_patch, depth_feature_maps_resized[0].shape[-2:], mask, label, center, neighbor_idx,
                             nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)

        if self.args.rm_zero_for_project:
            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib != 0, dim=1))[:, 0]]

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]

        if self.args.rm_zero_for_project:
            self.patch_lib = self.patch_lib[torch.nonzero(torch.all(self.patch_lib != 0, dim=1))[:, 0]]
            self.patch_lib = torch.cat((self.patch_lib, torch.zeros(1, self.patch_lib.shape[1])), 0)

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz,
                        center_idx, nonzero_patch_indices=None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())


class SIFTFeatures(Features):
    def __init__(self, args):
        super().__init__(args)
        self.args = args  # 存储args如果需要
        self.dsift = DenseSIFTDescriptor()

    def add_sample_to_mem_bank(self, sample):
        sample = sample[2]
        dsift_feat = self.dsift(sample[:, 0, :, :].unsqueeze(dim=1)).detach()
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        sift_depth_resized_maps = self.resize(self.average(dsift_feat))
        sift_patch = sift_depth_resized_maps.reshape(sift_depth_resized_maps.shape[1], -1).T
        self.patch_lib.append(sift_patch)


    def predict(self, sample, mask, label):


        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0],
                                                                                       unorganized_pc_no_zeros.contiguous())
        sample = sample[2]
        feature_maps = self.dsift(sample[:, 0, :, :].unsqueeze(dim=1)).detach()
        self.resize = torch.nn.AdaptiveAvgPool2d((28, 28))
        depth_feature_maps_resized = self.resize(self.average(feature_maps))
        siftpatch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        self.compute_s_s_map(siftpatch, depth_feature_maps_resized.shape[-2:], mask, label, center, neighbor_idx,
                             nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)
        self.mean = torch.mean(self.patch_lib)
        self.std = torch.std(self.patch_lib)
        self.patch_lib = (self.patch_lib - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]
    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        patch = (patch - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())


class HOGFeatures(Features):
    def add_sample_to_mem_bank(self, sample):
        sample = sample[2]
        hog_feature = hog(sample[0, 0, :, :], orientations=8, pixels_per_cell=(8, 8),
                          cells_per_block=(1, 1), visualize=False, feature_vector=False)
        hog_feature = hog_feature.reshape(hog_feature.shape[0],
                                          hog_feature.shape[1],
                                          hog_feature.shape[2] *
                                          hog_feature.shape[3] *
                                          hog_feature.shape[4])
        hog_feature = torch.tensor(hog_feature.squeeze()).permute(2, 0, 1).unsqueeze(dim=0)
        hog_depth_patch = hog_feature.reshape(hog_feature.shape[1], -1).T
        self.patch_lib.append(hog_depth_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0],
                                                                                       unorganized_pc_no_zeros.contiguous())
        sample = sample[2]
        hog_features = hog(sample[0, 0, :, :], orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                           visualize=False, feature_vector=False)
        hog_features = hog_features.reshape(hog_features.shape[0],
                                            hog_features.shape[1],
                                            hog_features.shape[2] * hog_features.shape[3] *
                                            hog_features.shape[4])
        depth_feature_maps_resized = torch.tensor(hog_features.squeeze()).permute(2, 0, 1).unsqueeze(dim=0)
        patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        self.compute_s_s_map(patch, depth_feature_maps_resized.shape[-2:], mask, label, center, neighbor_idx,
                             nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):

        self.patch_lib = torch.cat(self.patch_lib, 0)
        self.mean = torch.mean(self.patch_lib)
        self.std = torch.std(self.patch_lib)
        self.patch_lib = (self.patch_lib - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]
    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx, nonzero_patch_indices = None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        patch = (patch - self.mean)/self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())


class WideRGBFeatures(ADsFeatures):

    def add_sample_to_mem_bank(self, sample):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, _, _, center_idx, _ = self(sample[0], unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.patch_lib.append(rgb_patch)

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, _ = self(sample[0],
                                                                                       unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(rgb_patch, rgb_feature_maps[0].shape[-2:], mask, label, center, neighbor_idx,
                             nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0)
        self.mean = torch.mean(self.patch_lib)
        self.std = torch.std(self.patch_lib)
        self.patch_lib = (self.patch_lib - self.mean) / self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz,
                        center_idx, nonzero_patch_indices=None):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        patch = (patch - self.mean) / self.std

        # self.patch_lib = self.rgb_layernorm(self.patch_lib)
        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D)) + 1e-5))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(224, 224), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())


class DINOFPFHFusionTripleFeatures(Features):


    # 改进的fpfh
    def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
        """
        计算有序点云的FPFH特征并拼接三种不同体素大小的特征

        参数:
            organized_pc: torch.Tensor 形状为 [1, 3, H, W] 的有序点云张量
            voxel_size: float 用于计算特征半径的基础体素大小
            savefpfh_path: str 特征保存路径(当前未使用但保留接口)

        返回:
            torch.Tensor 形状为 [1, C, H, W] 的特征张量，其中C=3×FPFH维度(99)
        """
        # 输入检查和转换
        print("输入点云类型:", type(organized_pc))
        print("输入点云形状:", organized_pc.shape)

        # 转换为无序点云格式 [N, 3]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # [H,W,3] -> [N,3]
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_np)  # 自定义函数

        # 移除全零点(无效点)
        nonzero_mask = np.all(unorganized_pc != 0, axis=1)
        nonzero_indices = np.nonzero(nonzero_mask)[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

        # 创建Open3D点云对象
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        bbox = o3d_pc.get_axis_aligned_bounding_box()
        avg_size = np.mean(bbox.get_extent())
        voxel_scales = [0.05]
        all_fpfh_features = []
        for scale in voxel_scales:
            current_voxel_size = avg_size * scale
            radius_feature = current_voxel_size * 5  # 特征计算半径
            print(f"体素比例 {scale}: 体素大小 {current_voxel_size:.3f}, 特征半径: {radius_feature:.3f}")
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
            print(f"体素比例 {scale} 的FPFH特征形状: {fpfh.shape}")
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
        print("densities", densities, densities2, densities3, densities4)
        fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 4))
        print("fpfh.shape", fpfh.shape)
        print("fpfh_with_density.shape", fpfh_with_density.shape)
        fpfh_with_density[:, :-4] = fpfh
        # fpfh_with_density[:, -5] = densities5
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
        print("full_fpfhwithden_tensor ", full_fpfhwithden_tensor.shape)
        return full_fpfhwithden_tensor

    #PRL
    # def get_fpfh_features(self,organized_pc, voxel_size=0.05):
    #     # # Convert organized PointCloud to numpy array
    #     print("organized_pc.type", type(organized_pc))# <class 'torch.Tensor'>
    #     print("organized_pc.shape", organized_pc.shape)#torch.Size([1, 3, 224, 224])
    #     organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()#将原来的第一个维度（通常是批次维度）移动到最后一个位置
    #     ## Convert organized PointCloud to unorganized PointCloud
    #     unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    #     # # Find nonzero indices and remove zero elements
    #     nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    #     unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    #     # Create Open3D PointCloud object
    #     o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    #     # Estimate normals for the unorganized PointCloud
    #     radius_normal = voxel_size * 2
    #     o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    #     ## Compute Fast Point Feature Histogram (FPFH) features
    #     radius_feature = voxel_size * 5
    #     pcd_fpfh = o3d.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
    #     (radius=radius_feature, max_nn=100))
    #     fpfh = pcd_fpfh.data.T
    #     # Create full FPFH array and reshape it
    #     full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)#full_fpfh 被初始化为一个全零的数组，其形状为 (unorganized_pc.shape[0], fpfh.shape[1])。这个数组的行数与 unorganized_pc 的点数相同，列数与 fpfh 的特征数量相同。
    #     full_fpfh[nonzero_indices, :] = fpfh#这行代码将 fpfh 中的非零索引（nonzero_indices）对应的特征值填充到 full_fpfh 中的相应位置。
    #     print("full_fpfh[nonzero_indices, :] = fpfh,fpfh.shape",fpfh.shape)
    #     full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))#full_fpfh_reshaped 使用 reshape 将 full_fpfh 重塑为一个三维数组，形状为 (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1])。这意味着最终的数组将对应于一个有序点云，每个点都有其相应的特征。
    #     # # Convert FPFH features to a PyTorch tensor
    #     full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
    #     return full_fpfh_tensor







    def add_sample_to_mem_bank(self, sample, class_name=None):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps= self(sample[0])#简化了一下
        #rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
        #                                                                                             unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784*4, -1)
        self.patch_rgb_lib.append(rgb_patch)

        if self.args.asy_memory_bank is None or len(self.patch_xyz_lib) < self.args.asy_memory_bank:
            ############### FPFH PATCH ###############
            print("type(sample[1]", type(sample[1]))  # type(sample[1] <class 'torch.Tensor'>
            print("sample[1].shape", sample[1].shape)  # ([1, 3, 224, 224])
            fpfh_feature_maps = self.get_fpfh_features(
                sample[1])  # sample[1]torch.float32非常稀疏   fpfh_feature_mapstorch.float64,33,224,224
            print("sample[1].shape", sample[1].shape)
            fpfh_feature_maps_resized = self.resize(self.average(fpfh_feature_maps))  # 1,33,28,28 torch.float64
            fpfh_patch = fpfh_feature_maps_resized.reshape(fpfh_feature_maps_resized.shape[1],
                                                           -1).T  # torch.float64784 33
            ############### END FPFH PATCH ###############
            self.patch_xyz_lib.append(fpfh_patch)
            self.fusion = FeatureFusionBlock(37, 768, mlp_ratio=4.)
            if FUSION_BLOCK:
                with torch.no_grad():
                    fusion_patch = self.fusion.feature_fusion(fpfh_patch.unsqueeze(0), rgb_patch_resize.unsqueeze(0))
                fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
            else:
                fusion_patch = torch.cat([fpfh_patch, rgb_patch_resize], dim=1)
            self.patch_fusion_lib.append(fusion_patch)

        if class_name is not None:
            torch.save(fusion_patch, os.path.join(self.args.save_feature_path, class_name + str(self.ins_id) + '.pt'))
            self.ins_id += 1

    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps  = self(sample[0] )



        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch_resize = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch_resize = rgb_patch_resize.reshape(rgb_patch.shape[1], -1).T

        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(fpfh_patch.unsqueeze(0), rgb_patch_resize.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([fpfh_patch, rgb_patch_resize], dim=1)

        # self.compute_s_s_map(fpfh_patch, rgb_patch, fusion_patch, depth_feature_maps_resized[0].shape[-2:], mask, label,
        #                      center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        self.compute_s_s_map(fpfh_patch, rgb_patch, fusion_patch, depth_feature_maps_resized[0].shape[-2:], mask, label)

    def add_sample_to_late_fusion_mem_bank(self, sample):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        # rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
        #                                                                                              unorganized_pc_no_zeros.contiguous())
        rgb_feature_maps = self(sample[0])
        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch_resize = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch_resize = rgb_patch_resize.reshape(rgb_patch.shape[1], -1).T
        if FUSION_BLOCK:
            with torch.no_grad():
                fusion_patch = self.fusion.feature_fusion(fpfh_patch.unsqueeze(0), rgb_patch_resize.unsqueeze(0))
            fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        else:
            fusion_patch = torch.cat([fpfh_patch, rgb_patch_resize], dim=1)

        # 3D dist
        xyz_patch = (fpfh_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean) / self.fusion_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size,
                                                             modal='fusion')
        s = torch.tensor(
            [[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb, self.args.fusion_s_lambda * s_fusion]])
        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb,
                           self.args.fusion_smap_lambda * s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def run_coreset(self):  # class TripleFeatures(Features):
        #
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean) / self.fusion_std



        print("self.patch_xyz_lib.shape", self.patch_xyz_lib.shape)
        print("self.patch_rgb_lib.shape", self.patch_rgb_lib.shape)
        print("self.patch_fusion_lib.shape", self.patch_fusion_lib.shape)

        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_xyz_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_rgb_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_fusion_lib,
                                                            n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]
        print("下采样后self.patch_xyz_lib.shape", self.patch_xyz_lib.shape)
        print("下采样后self.patch_rgb_lib.shape", self.patch_rgb_lib.shape)
        print("下采样后self.patch_fusion_lib.shape", self.patch_fusion_lib.shape)



        self.patch_xyz_lib = self.patch_xyz_lib[torch.nonzero(torch.all(self.patch_xyz_lib != 0, dim=1))[:, 0]]
        self.patch_xyz_lib = torch.cat((self.patch_xyz_lib, torch.zeros(1, self.patch_xyz_lib.shape[1])), 0)  #
        print("torch.zeros(1, self.patch_xyz_lib.shape[1]))", self.patch_xyz_lib.shape)
        # torch.zeros(1, self.patch_xyz_lib.shape[1])) torch.Size([76519, 1152])

    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist     # 3D距离计算前的标准化处理
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean) / self.fusion_std
        # # 计算当前数据块与库中数据块的欧氏距离
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)
        # 计算特征图的大小（假设特征图是正方形的）
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 分别计算每种模态（xyz, rgb, fusion）的显著性得分和显著性图
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size,
                                                             modal='fusion')
        # 根据配置参数（lambda值）融合不同模态的显著性得分
        s = torch.tensor(
            [[self.args.xyz_s_lambda * s_xyz, self.args.rgb_s_lambda * s_rgb, self.args.fusion_s_lambda * s_fusion]])
        print("s.shape", s.shape)
        # torch.Size([1, 3])
        # 同样地，融合不同模态的显著性图
        s_map = torch.cat([self.args.xyz_smap_lambda * s_map_xyz, self.args.rgb_smap_lambda * s_map_rgb,
                           self.args.fusion_smap_lambda * s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)
        print(" s_map.shape", s_map.shape)
        # # 使用特定的融合器对显著性得分和显著性图进行评分
        s = torch.tensor(self.detect_fuser.score_samples(s))
        #    # 调整显著性图的形状以匹配图像尺寸
        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))
        ## 保存结果以便后续分析或评估
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
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal == 'xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal == 'rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal == 'xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)
        elif modal == 'rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1)
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear',
                                                align_corners=False)
        s_map = self.blur(s_map)

        return s, s_map
