"""
 logic based on https://github.com/rvorias/ind_knn_ad
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn import random_projection
from sklearn import linear_model
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# from timm.models.layers import DropPath, trunc_normal_
#from pointnet2_ops import pointnet2_utils
#from knn_cuda import KNN

from utils.utils import KNNGaussianBlur
from utils.utils import set_seeds
from utils.au_pro_util import calculate_au_pro

from models.pointnet2_utils import interpolating_points
from models.feature_fusion import FeatureFusionBlock
from models.models import Model
from models.ssl import SSLBlock

class Features(torch.nn.Module):

    def __init__(self, args, image_size=224, f_coreset=0.1, coreset_eps=0.9):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(
                                device=self.device, 
                                rgb_backbone_name=args.rgb_backbone_name, 
                                xyz_backbone_name=args.xyz_backbone_name, 
                                group_size = args.group_size, 
                                num_group=args.num_group
                                )
        self.deep_feature_extractor.to(self.device)

        self.args = args
        self.image_size = args.img_size
        self.f_coreset = args.f_coreset
        self.coreset_eps = args.coreset_eps
        
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_xyz_lib = []
        self.patch_rgb_lib = []
        self.patch_fusion_lib = []
        self.patch_lib = []
        self.random_state = args.random_state

        self.xyz_dim = 0
        self.rgb_dim = 0

        self.xyz_mean=0
        self.xyz_std=0
        self.rgb_mean=0
        self.rgb_std=0
        self.fusion_mean=0
        self.fusion_std=0

        self.average = torch.nn.AvgPool2d(3, stride=1) # torch.nn.AvgPool2d(1, stride=1) #
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))
        self.resize2 = torch.nn.AdaptiveAvgPool2d((56, 56))

        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0
        self.ins_id = 0
        self.rgb_layernorm = torch.nn.LayerNorm(768, elementwise_affine=False)

        if self.args.use_uff:
            #self.fusion = FeatureFusionBlock(1152, 768, mlp_ratio=4.)
            self.fusion = FeatureFusionBlock(37, 768, mlp_ratio=4.)
            #print("self.fusion = FeatureFusionBlock(1152, 768, mlp_ratio=4.)",self.fusion)
            print("args.fusion_module_path")
            ckpt = torch.load(args.fusion_module_path)['model']

            incompatible = self.fusion.load_state_dict(ckpt, strict=False)

            print('[Fusion Block]', incompatible)
        if self.args.use_ssl:
            self.fusion = SSLBlock(1152, 768, mlp_ratio=4.)
            ckpt = torch.load(args.ssl_module_path)['model']
            incompatible = self.fusion.load_state_dict(ckpt, strict=False)
            print('[Fusion Block]', incompatible)

        self.detect_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=args.ocsvm_nu,  max_iter=args.ocsvm_maxiter)
        self.seg_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=args.ocsvm_nu,  max_iter=args.ocsvm_maxiter)

        self.s_lib = []
        self.s_map_lib = []

        self.projection_matrix = None

    def __call__(self, rgb):
        # Extract the desired feature maps using the backbone model.
        rgb = rgb.to(self.device)
        #xyz = xyz.to(self.device)
        with torch.no_grad():
            rgb_feature_maps  = self.deep_feature_extractor(rgb)
        # interpolate = True
        # if interpolate:
        #     interpolated_feature_maps = interpolating_points(xyz, center.permute(0,2,1), xyz_feature_maps).to("cpu")
        #xyz_feature_maps = [fmap.to("cpu") for fmap in [xyz_feature_maps]]
        rgb_feature_maps = [fmap.to("cpu") for fmap in [rgb_feature_maps]]
        #rgb_feature_maps = [fmap.to("cpu") for fmap in rgb_feature_maps]
        return rgb_feature_maps

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def add_sample_to_late_fusion_mem_bank(self, sample):
        raise NotImplementedError

    def interpolate_points(self, rgb, xyz):
        with torch.no_grad():
            rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(rgb, xyz)
        return xyz_feature_maps, center, xyz
    
    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz, center_idx):
        raise NotImplementedError
    
    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):
        raise NotImplementedError
    
    def run_coreset(self):
        raise NotImplementedError

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)
        # # print(type(self.image_labels), type(self.image_preds))#<class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print("self.image_labels[:10]",self.image_labels[:10])  # 输出前10个样本self.image_labels[:10] [[1]
        # print("self.image_preds[:10]",self.image_preds[:10])  # 输出前10个预测值
        self.image_labels = np.array(self.image_labels, dtype=np.float64)#
        self.image_preds = np.array(self.image_preds, dtype=np.float64)
        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def save_prediction_maps(self, output_path, rgb_path, save_num=5):
        for i in range(max(save_num, len(self.predictions))):            
            # fig = plt.figure(dpi=300)
            fig = plt.figure()
        
            ax3 = fig.add_subplot(1,3,1)
            gt = plt.imread(rgb_path[i][0])    
            ax3.imshow(gt)

            ax2 = fig.add_subplot(1,3,2)
            im2 = ax2.imshow(self.gts[i], cmap=plt.cm.gray)
            
            ax = fig.add_subplot(1,3,3)
            im = ax.imshow(self.predictions[i], cmap=plt.cm.jet)
            
            class_dir = os.path.join(output_path, rgb_path[i][0].split('/')[-5])
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-3])
            if not os.path.exists(ad_dir):
                os.mkdir(ad_dir)
            
            plt.savefig(os.path.join(ad_dir,  str(self.image_preds[i]) + '_pred_' + rgb_path[i][0].split('/')[-1] + '.jpg'))
            
    def run_late_fusion(self):
        self.s_lib = torch.cat(self.s_lib, 0)
        self.s_map_lib = torch.cat(self.s_map_lib, 0)
        self.detect_fuser.fit(self.s_lib)
        self.seg_fuser.fit(self.s_map_lib)



#正交矩阵
    def _next_power_of_two(self, n):
        """获取下一个2的幂"""
        return 1 << (n - 1).bit_length()

    def _fast_orthogonal_matrix(self, d, target_dim):
        """快速生成近似正交投影矩阵的核心方法"""
        # 1. 随机符号对角矩阵 (±1)
        print(f"DEBUG: d_original = {d}, target_dim = {target_dim}")  # Add debug print
        torch.manual_seed(self.random_state if self.random_state else np.random.randint(1e6))
        D = torch.diag(torch.randint(0, 2, (d,)) * 2 - 1).float()
        # 2. 判断维度是否为2的幂
        if (d & (d - 1)) == 0:  # d是2的幂
            # 使用哈达玛矩阵的快速构造
            H = self._fast_hadamard_like(d)
            # 组合: D -> H -> 随机子采样
            projection = D @ H
        else:
            # 混合策略: D -> 部分哈达玛 -> 随机傅立叶
            pow2 = self._next_power_of_two(d)
            H_partial = self._partial_hadamard(d, pow2)
            projection = D @ H_partial
            # 补充随机傅立叶部分
            if target_dim < d // 2:
                F = self._random_fourier(d - pow2)
                projection = torch.cat([projection, F], dim=1)[:, :d]

        # 3. 随机列选择 (Johnson-Lindenstrauss的核心)
        if target_dim < d:
            indices = torch.randperm(d)[:target_dim]
            projection = projection[:, indices]

        # 4. 归一化以保持近似正交性
        projection = projection / torch.sqrt(torch.tensor(target_dim, dtype=torch.float32))

        return projection

    def _fast_hadamard_like(self, n):
        """快速构造哈达玛风格的变换矩阵"""
        H = torch.eye(n)
        stride = 1
        while stride < n:
            for i in range(0, n, 2 * stride):
                for j in range(stride):
                    a = H[:, i + j].clone()
                    b = H[:, i + j + stride].clone()
                    H[:, i + j] = a + b
                    H[:, i + j + stride] = a - b
            stride *= 2
        return H / torch.sqrt(torch.tensor(n, dtype=torch.float32))

    def _partial_hadamard(self, d, pow2):
        """构造部分哈达玛矩阵"""
        H_full = self._fast_hadamard_like(pow2)
        # 随机选择d行
        indices = torch.randperm(pow2)[:d]
        return H_full[indices, :]

    def _random_fourier(self, n):
        """随机傅立叶特征"""
        W = torch.randn(n, n) / np.sqrt(n)
        # 模拟傅立叶变换的实部
        F_real = torch.cos(W @ torch.randn(n, 1))
        F_imag = torch.sin(W @ torch.randn(n, 1))
        return (F_real + F_imag) / np.sqrt(2)

    def fit_transform(self, X):
        """关键API：与SparseRandomProjection保持一致"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        target_dim=64
        n_samples, d_original = X.shape

        # 根据JL引理确定目标维度 (与eps参数对应)
        # 这里保持与原来类似的维度缩减逻辑
        #target_dim = max(1, int(np.ceil(np.log(n_samples) / (self.eps ** 2))))
        #target_dim = min(target_dim, d_original // 2)  # 安全限制

        # 生成投影矩阵
        if self.projection_matrix is None:
            self.projection_matrix = self._fast_orthogonal_matrix(d_original, target_dim)

        # 执行投影
        X_projected = X @ self.projection_matrix

        return X_projected.numpy()


    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.7, float16=True, force_cpu=False):#eps=0.90
        print(f"  don not Use ogpt Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps, random_state=self.random_state)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))#z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        # try:
        #     # 使用新的快速正交投影
        #     transformer = self.FastOrthogonalProjection(eps=eps, random_state=self.random_state)
        #     z_lib = torch.tensor(transformer.fit_transform(z_lib))
        #     # 确保输入格式正确
        #     if isinstance(z_lib, torch.Tensor):
        #         z_lib_np = z_lib.cpu().numpy()
        #     else:
        #         z_lib_np = z_lib
        #     z_lib_projected = self.fit_transform(z_lib_np)
        #     z_lib = torch.tensor(z_lib_projected)
        #     print(f"   DONE. Transformed dim = {z_lib.shape}. (Orthogonal Projection)")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")
        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")
        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step
            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)
