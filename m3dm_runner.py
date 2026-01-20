import torch
from tqdm import tqdm
import os

from feature_extractors import multiple_features
from feature_extractors import selfmultiple_features
from dataset import get_data_loader

class M3DM():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        self.count = args.max_sample
        if args.method_name == 'DINO':
            self.methods = {
                "DINO": multiple_features.RGBFeatures(args),
            }
        elif args.method_name == 'SSLspot':
            self.methods = {
                "SSLspot": selfmultiple_features.SSLspot(args),
            }
        elif args.method_name == 'Point_MAE':
            self.methods = {
                "Point_MAE": multiple_features.PointFeatures(args),
            }
        elif args.method_name == 'Fusion':
            self.methods = {
                "Fusion": multiple_features.FusionFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE':
            self.methods = {
                "DINO+Point_MAE": multiple_features.DoubleRGBPointFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE+add':
            self.methods = {
                "DINO+Point_MAE+add": multiple_features.DoubleRGBPointFeatures_add(args),
            }
        elif args.method_name == 'DINO+Point_MAE+Fusion':
            self.methods = {
                "DINO+Point_MAE+Fusion": multiple_features.TripleFeatures(args),
            }
        elif args.method_name == 'DINO+FPFH+Fusion':
            self.methods = {
                "DINO+FPFH+Fusion": selfmultiple_features.DINOFPFHFusionTripleFeatures(args),
            }
        elif args.method_name == 'DINO+FPFH+add':#best
            self.methods = {
                "DINO+FPFH+add": selfmultiple_features.DoubleRGBFPFHFeatures_add(args),
            }
        elif args.method_name == 'DINO+FPFH':
            self.methods = {
                "DINO+FPFH":  selfmultiple_features.DoubleRGBFPFHFeatures(args),
            }
        elif args.method_name == 'FPFH':
            self.methods = {
                "FPFH":  selfmultiple_features.FPFHFeatures(args),
            }
        elif args.method_name == 'SIFT':
            self.methods = {
                "SIFT":  selfmultiple_features.SIFTFeatures(args),
            }
        elif args.method_name == 'HOG':
            self.methods = {
                "HOG":  selfmultiple_features.HOGFeatures(args),
            }
        elif args.method_name == 'WideRGB':
            self.methods = {
                "WideRGB":  selfmultiple_features.WideRGBFeatures(args),
            }

    def _clear_methods_memory(self):
            """清理所有方法对象内部累积的大型列表和缓存，防止内存泄漏。"""
            for method_name, method in self.methods.items():
                if hasattr(method, 'patch_lib'):
                    # 清空特征库列表，并尝试释放内存
                    method.patch_lib.clear()
                    # 将列表本身设置为一个小列表，触发Python内存回收
                    method.patch_lib = []
                if hasattr(method, 'rgb_feature_lib'):  # 如果还有其他库，一并清理
                    method.rgb_feature_lib.clear()
                    method.rgb_feature_lib = []
                if hasattr(method, 'xyz_feature_lib'):
                    method.xyz_feature_lib.clear()
                    method.xyz_feature_lib = []
                # 提示：检查你的method类，将所有类似的 `.append` 列表都清理掉

                # 如果方法有CUDA缓存，也一并清理（例如特征提取网络可能缓存了中间结果）
                if torch.cuda.is_available():
                    # 这里假设method本身可能有缓存，或者其内部的模型有缓存
                    # 更激进的做法：调用其内部模型的 `empty_cache` (如果有)
                    pass
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)

        flag = 0
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            # countm=0
            for method in self.methods.values():
                # countm+=1
                # print("count method",countm)#始终为1
                #print("method",method)
                if self.args.save_feature:
                    method.add_sample_to_mem_bank(sample, class_name=class_name)
                else:
                    method.add_sample_to_mem_bank(sample)
                flag += 1
            if flag > self.count:
                flag = 0
                break
                
        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()
            

        if self.args.memory_bank == 'multiple':    
            flag = 0
            for sample, _ in tqdm(train_loader, desc=f'Running late fusion for {method_name} on class {class_name}..'):
                for method_name, method in self.methods.items():
                    method.add_sample_to_late_fusion_mem_bank(sample)
                    flag += 1
                if flag > self.count:
                    flag = 0
                    break
        
            for method_name, method in self.methods.items():
                print(f'\n\nTraining Dicision Layer Fusion for {method_name} on class {class_name}...')
                method.run_late_fusion()
        self._clear_methods_memory()



    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)
        path_list = []
        with torch.no_grad():
        
            for sample, mask, label, rgb_path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                for method in self.methods.values():
                    method.predict(sample, mask, label)
                    path_list.append(rgb_path)
                        

        for method_name, method in self.methods.items():
            method.calculate_metrics()
            image_rocaucs[method_name] = round(method.image_rocauc, 3)
            pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
            au_pros[method_name] = round(method.au_pro, 3)
            print(
                f'Class: {class_name}, {method_name} Image ROCAUC: {method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {method.pixel_rocauc:.3f}, {method_name} AU-PRO: {method.au_pro:.3f}')
            # if self.args.save_preds:
            #     method.save_prediction_maps('./noiseconwithoutproMDAE_o', path_list)
              #  method.save_prediction_maps('./pred_maps_m3dm', path_list)#
            # ========== 【新增】在fit函数结束前，清理所有method的内部缓存 ==========
        self._clear_methods_memory()
            # =================================================================
        return image_rocaucs, pixel_rocaucs, au_pros
