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
        print(f"log_probabilities: {log_probabilities}")
        entropy = -torch.sum(probabilities * log_probabilities)
        return entropy
    def calculate_gating_weights(self, encoder_output_1, encoder_output_2):
        entropy_1 = np.log(self.calculate_entropy(encoder_output_1))
        entropy_2 = np.log(self.calculate_entropy(encoder_output_2))
        print(f"Entropy 1: {entropy_1}, Entropy 2: {entropy_2}")
        sum_weights = entropy_1 + entropy_2
        print(f"sum_weights: {sum_weights}")
        entropy_1 /= sum_weights
        entropy_2 /= sum_weights
        print(f"Entropy 1: {entropy_1}, Entropy 2: {entropy_2}")
        return entropy_1, entropy_2
    def l2_normalize(self,features):
        norms = np.linalg.norm(features, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return features / norms


    # def _torch_to_o3d(t: torch.Tensor) -> o3d.core.Tensor:
    #     # Open3D 支持从带 __dlpack__ 的外部 tensor 直接零拷贝导入 :contentReference[oaicite:3]{index=3}
    #     return o3d.core.Tensor.from_dlpack(t.contiguous())
    #
    # def _o3d_to_torch(t: o3d.core.Tensor) -> torch.Tensor:
    #     # Open3D Tensor 支持 __dlpack__，torch 可直接 from_dlpack 零拷贝 :contentReference[oaicite:4]{index=4}
    #     return torch.utils.dlpack.from_dlpack(t)
    #
    # def _fixed_radius_counts(nns: o3d.core.nns.NearestNeighborSearch,
    #                          queries: o3d.core.Tensor,
    #                          radius: float,
    #                          num_queries: int,
    #                          device: torch.device) -> torch.Tensor:
    #     # fixed_radius_search 返回 (indices, splits, distances)（文档）但示例里顺序有歧义；
    #     # 这里用 shape 规则自动识别 splits（splits.shape == [num_queries + 1]）:contentReference[oaicite:5]{index=5}
    #     a, b, c = nns.fixed_radius_search(queries, radius=radius, sort=False)
    #
    #     # 找 splits
    #     splits = None
    #     for x in (a, b, c):
    #         if len(x.shape) == 1 and int(x.shape[0]) == num_queries + 1:
    #             splits = x
    #             break
    #     if splits is None:
    #         raise RuntimeError("Cannot identify 'splits' tensor from fixed_radius_search outputs.")
    #
    #     splits_t = _o3d_to_torch(splits).to(device)
    #     counts = (splits_t[1:] - splits_t[:-1]).to(torch.float32)  # [N]
    #     return counts
    # # ===== 你的函数（GPU版）=====
    # def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
    #     """
    #     organized_pc: torch.Tensor, 常见形状 [1,3,H,W] 或 [3,H,W] 或 [H,W,3]
    #     return: [1, C, H, W]，其中 C = 33*len(voxel_scales) + 4
    #     """
    #     assert isinstance(organized_pc, torch.Tensor)
    #
    #     # ---- 选择 Open3D device（CUDA 优先） ----
    #     use_cuda = organized_pc.is_cuda and o3d.core.cuda.is_available()
    #     o3d_device = o3d.core.Device("cuda:0" if use_cuda else "cpu:0")
    #
    #     # ---- 组织点云 -> [H,W,3] -> [N,3]（全 torch）----
    #     pc = organized_pc
    #     if pc.ndim == 4:
    #         pc = pc.squeeze(0)
    #     if pc.shape[0] == 3:
    #         pc_hw3 = pc.permute(1, 2, 0)  # [H,W,3]
    #     elif pc.shape[-1] == 3:
    #         pc_hw3 = pc  # [H,W,3]
    #     else:
    #         raise ValueError(f"Unexpected organized_pc shape: {tuple(organized_pc.shape)}")
    #
    #     H, W, _ = pc_hw3.shape
    #     unorganized = pc_hw3.reshape(-1, 3)  # [N,3]
    #
    #     # 移除全零点（无效点）
    #     nonzero_mask = ~torch.all(unorganized == 0, dim=1)  # [N]
    #     points = unorganized[nonzero_mask].to(torch.float32)  # [Nnz,3]
    #
    #     # bbox avg_size（torch 算，避免 legacy Open3D）
    #     extent = points.max(dim=0).values - points.min(dim=0).values
    #     avg_size = float(extent.mean().item())
    #
    #     # ---- 构建 Open3D Tensor PointCloud（零拷贝 or 必要拷贝）----
    #     pts_for_o3d = points if use_cuda else points.detach().cpu()
    #     pcd = o3d.t.geometry.PointCloud(o3d_device)  # 支持指定 device 创建 :contentReference[oaicite:6]{index=6}
    #     pcd.point.positions = _torch_to_o3d(pts_for_o3d)
    #
    #     # ---- 多尺度 FPFH（你原代码只有 [0.05]，这里保留结构）----
    #     voxel_scales = [0.05]
    #     fpfh_list = []
    #     for scale in voxel_scales:
    #         current_voxel_size = avg_size * scale
    #         radius_feature = current_voxel_size * 5.0
    #
    #         # Tensor API normals：建议同时给 radius + max_nn，CUDA 更重要 :contentReference[oaicite:7]{index=7}
    #         pcd.estimate_normals(max_nn=30, radius=current_voxel_size)
    #
    #         # Tensor API FPFH：同时给 radius + max_nn -> hybrid search（推荐）:contentReference[oaicite:8]{index=8}
    #         fpfh_o3d = o3d.t.pipelines.registration.compute_fpfh_feature(
    #             pcd, radius=radius_feature, max_nn=100
    #         )  # (Nnz,33) open3d.core.Tensor
    #         fpfh_t = _o3d_to_torch(fpfh_o3d).to(organized_pc.device).contiguous()  # torch (Nnz,33)
    #         fpfh_list.append(fpfh_t)
    #
    #     fpfh = torch.cat(fpfh_list, dim=1)  # (Nnz, 33*scales)
    #
    #     # ---- densities：GPU 半径邻域计数 ----
    #     # 用 open3d.core.nns.NearestNeighborSearch（支持 fixed-radius search / splits）:contentReference[oaicite:9]{index=9}
    #     pos_o3d = pcd.point.positions
    #     nns = o3d.core.nns.NearestNeighborSearch(pos_o3d)
    #
    #     radii = [
    #         voxel_size * 0.03,
    #         voxel_size * 0.07,
    #         voxel_size * 0.11,
    #         voxel_size * 0.15,
    #     ]
    #     max_r = max(radii)
    #
    #     # GPU fixed radius index 需要给 radius 初始化 :contentReference[oaicite:10]{index=10}
    #     nns.fixed_radius_index(radius=max_r)
    #
    #     Nnz = int(pos_o3d.shape[0])
    #     # queries 就用自身点
    #     queries = pos_o3d
    #
    #     d1 = _fixed_radius_counts(nns, queries, radii[0], Nnz, organized_pc.device)
    #     d2 = _fixed_radius_counts(nns, queries, radii[1], Nnz, organized_pc.device)
    #     d3 = _fixed_radius_counts(nns, queries, radii[2], Nnz, organized_pc.device)
    #     d4 = _fixed_radius_counts(nns, queries, radii[3], Nnz, organized_pc.device)
    #
    #     # 拼接：[-4]=dens4, [-3]=dens3, [-2]=dens2, [-1]=dens1（对齐你原逻辑）
    #     fpfh_with_density = torch.cat(
    #         [fpfh, d4[:, None], d3[:, None], d2[:, None], d1[:, None]],
    #         dim=1
    #     )  # (Nnz, 33*scales + 4)
    #
    #     # ---- 填回原 H*W 的位置（无效点=0）----
    #     N = unorganized.shape[0]
    #     C = fpfh_with_density.shape[1]
    #     full = torch.zeros((N, C), device=organized_pc.device, dtype=fpfh_with_density.dtype)
    #     full[nonzero_mask] = fpfh_with_density
    #
    #     out = full.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,C,H,W]
    #     return out

       # 改进的fpfh
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

    # ###原PRletters的
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



    # # 改进的fpfhEye
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
    #     voxel_scales = [0.06]
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
    #     #densities5 = np.zeros(len(points))
    #     for i in range(len(points)):
    #         [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.05)
    #         densities[i] = k  # 邻域内点的数量作为密度
    #         [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.01)
    #         densities2[i] = k2  # 邻域内点的数量作为密度
    #         [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
    #         densities3[i] = k3  # 邻域内点的数量作为密度
    #         [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.2)
    #         densities4[i] = k4
    #         # [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.07)
    #         # densities[i] = k  # 邻域内点的数量作为密度
    #         # [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.13)
    #         # densities2[i] = k2  # 邻域内点的数量作为密度
    #         # [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.19)
    #         # densities3[i] = k3  # 邻域内点的数量作为密度
    #         # [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.25)
    #         # densities4[i] = k4
    #
    #         # [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.04)
    #         # densities[i] = k  # 邻域内点的数量作为密度
    #         # [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.06)
    #         # densities2[i] = k2  # 邻域内点的数量作为密度
    #         # [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.08)
    #         # densities3[i] = k3  # 邻域内点的数量作为密度
    #         # [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.1)
    #         # densities4[i] = k4  #
    #         # [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.04)
    #         # densities[i] = k  # 邻域内点的数量作为密度
    #         # [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.06)
    #         # densities2[i] = k2  # 邻域内点的数量作为密度
    #         # [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.08)
    #         # densities3[i] = k3  # 邻域内点的数量作为密度
    #         # [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.1)
    #         # densities4[i] = k4
    #         # [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.03)
    #         # densities[i] = k  # 邻域内点的数量作为密度
    #         # [k2, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.07)
    #         # densities2[i] = k2  # 邻域内点的数量作为密度
    #         # [k3, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.11)
    #         # densities3[i] = k3  # 邻域内点的数量作为密度
    #         # [k4, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], voxel_size * 0.15)
    #         # densities4[i] = k4
    #
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






    # #
    # #
    # #
    # #
    # #
    # #
    # #
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
    #     # print("densities", densities, densities2, densities3, densities4)
    #     # fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 4))
    #     # print("fpfh.shape", fpfh.shape)
    #     # print("fpfh_with_density.shape", fpfh_with_density.shape)
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
        rgb_feature_maps = self(sample[0])
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
            print("saved patch", patch.shape)
            self.ins_id += 1
        self.patch_xyz_lib.append(fpfh_patch)
        self.patch_rgb_lib.append(rgb_patch)


    def predict(self, sample, mask, label):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        rgb_feature_maps = self(sample[0])

        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T


        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        self.compute_s_s_map(fpfh_patch, rgb_patch, depth_feature_maps_resized[0].shape[-2:], mask, label, nonzero_indices, unorganized_pc_no_zeros.contiguous())

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


    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, nonzero_indices, xyz):
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
        print("xyz_conf, rgb_conf,s",xyz_conf, rgb_conf,s)
        s_map = (xyz_conf *s_map_xyz + s_map_rgb* rgb_conf)
        s_map = s_map.view(1, self.image_size, self.image_size)
        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())



    # def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
    #     '''
    #     center: point group center position
    #     neighbour_idx: each group point index
    #     nonzero_indices: point indices of original point clouds
    #     xyz: nonzero point clouds
    #     '''
    #     # 2D dist
    #     xyz_patch = (xyz_patch - self.xyz_mean)/self.xyz_std
    #     rgb_patch = (rgb_patch - self.rgb_mean)/self.rgb_std
    #     dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
    #     dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
    #     print("type(self.patch_xyz_lib)",type(self.patch_xyz_lib))
    #     rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
    #     xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
    #     s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
    #     s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
    #
    #     xyz_conf,rgb_conf = self.calculate_gating_weights(xyz_patch, rgb_patch)
    #
    #     s = (xyz_conf * s_xyz + s_rgb* rgb_conf)
    #     print("xyz_conf, rgb_conf,s",xyz_conf, rgb_conf,s)
    #     s_map = (xyz_conf *s_map_xyz + s_map_rgb* rgb_conf)
    #     s_map = s_map.view(1, self.image_size, self.image_size)
    #     self.image_preds.append(s.numpy())
    #     self.image_labels.append(label)
    #     self.pixel_preds.extend(s_map.flatten().numpy())
    #     self.pixel_labels.extend(mask.flatten().numpy())
    #     self.predictions.append(s_map.detach().cpu().squeeze().numpy())
    #     self.gts.append(mask.detach().cpu().squeeze().numpy())

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

