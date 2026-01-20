import argparse
from m3dm_runner import M3DM
#from m3dm_runnerTrans import M3DMTrans
#from m3dmmetric_runner import M3DMmetric
from m3dm_runnerlate import M3DMlate
from dataset import eyecandies_classes, mvtec3d_classes
import pandas as pd
from SSLspot_runner import SSLspot
#from metric_learn import Covariance,RCA
# from m3dmrca_runner import M3DMRCA
# from m3dmpca_runner import M3DMPCA
# from m3dmdic_runner import M3DMDIC
import psutil
import os


def get_memory_comparison():
    """获取系统内存和进程内存对比"""
    process = psutil.Process(os.getpid())
    system_memory = psutil.virtual_memory()

    print("=== 系统内存 vs 进程内存 ===")
    print(f"系统总内存: {system_memory.total / 1024 ** 3:.2f} GB")
    print(f"系统可用内存: {system_memory.available / 1024 ** 3:.2f} GB")
    print(f"系统使用率: {system_memory.percent}%")

    proc_mem = process.memory_info().rss
    print(f"进程占用内存: {proc_mem / 1024 ** 2:.2f} MB")
    print(f"进程占系统比例: {proc_mem / system_memory.total * 100:.4f}%")




def run_3d_ads(args):
    if args.dataset_type=='eyecandies':
        classes = eyecandies_classes()
    elif args.dataset_type=='mvtec3d':
        classes = mvtec3d_classes()
    METHOD_NAMES = [args.method_name]
    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for cls in classes:
        if args.method_name == 'SSLspot':
            model=SSLspot(args)
            model.train_model(cls)
            image_rocaucs, pixel_rocaucs, au_pros = model.evaluate(cls)
        elif args.method_name =='DINO+FPFH+add+late':
            model=M3DMlate(args)#实例化模型
            model.fit(cls)#特征提取与构建模板库
            image_rocaucs, pixel_rocaucs, au_pros = model.evaluate(cls)
        else:
            model = M3DM(args)
            get_memory_comparison()
            model.fit(cls)
            print("fit")
            get_memory_comparison()
            image_rocaucs, pixel_rocaucs, au_pros = model.evaluate(cls)
            print("evaluate")
            get_memory_comparison()
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)
        print(f"\nFinished running on class {cls}")
        get_memory_comparison()
        # 1. 删除模型实例（这是内存大户）
        del model
        # 2. 删除评估结果变量（如果它们很大）
        del image_rocaucs, pixel_rocaucs, au_pros
        print("################################################################################\n\n")
        get_memory_comparison()
    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)
    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))
    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))
    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))
    with open("results/image_rocauc_results.md", "a") as tf:
        tf.write(image_rocaucs_df.to_markdown(index=False))
    with open("results/pixel_rocauc_results.md", "a") as tf:
        tf.write(pixel_rocaucs_df.to_markdown(index=False))
    with open("results/aupro_results.md", "a") as tf:
        tf.write(au_pros_df.to_markdown(index=False))
    # 合并所有DataFrame
    combined_df = pd.concat([image_rocaucs_df, pixel_rocaucs_df, au_pros_df], ignore_index=True)
    # 将合并后的DataFrame保存为CSV文件
    combined_df.to_csv('m3dm_5_DoubleRGBFPFHFeatures_add_results.csv', index=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--method_name', default='DINO+Point_MAE+Fusion', type=str,
                        choices=['DINO', 'SSLspot','Point_MAE', 'Fusion', 'DINO+Point_MAE', 'DINO+Point_MAE+Fusion', 'DINO+Point_MAE+add','DINO+FPFH+add+PCA','DINO+FPFH+add','DINO+FPFH+add+late','DINO+FPFH','DINO+FPFH+add+metric','DINO+FPFH+add+RCA','DINO+FPFH+add+DIC','DINO+FPFH+add+Trans','FPFH','SIFT','HOG','WideRGB','DINO+FPFH+Fusion'],
                        help='Anomaly detection modal name.')
    parser.add_argument('--max_sample', default=400, type=int,
                        help='Max sample number.')
    parser.add_argument('--memory_bank', default='single', type=str,
                        choices=["multiple", "single"],
                        help='memory bank mode: "multiple", "single".')
    parser.add_argument('--rgb_backbone_name', default='vit_base_patch8_224_dino', type=str, 
                        choices=['vit_base_patch8_224_dino', 'vit_base_patch8_224', 'vit_base_patch8_224_in21k', 'vit_small_patch8_224_dino'],
                        help='Timm checkpoints name of RGB backbone.')
    parser.add_argument('--xyz_backbone_name', default='Point_MAE', type=str, choices=['Point_MAE', 'Point_Bert'],
                        help='Checkpoints name of RGB backbone[Point_MAE, Point_Bert].')
    parser.add_argument('--fusion_module_path', default='/home/zgp/Documents/M3DM_5_3080/uff_pretrain.pth', type=str,
                        help='Checkpoints for fusion module.')
    parser.add_argument('--save_feature', default=False, action='store_true',
                        help='Save feature for training fusion block.')
    parser.add_argument('--use_uff', default=False, action='store_true',
                        help='Use UFF module.')
    parser.add_argument('--use_trans', default=True, action='store_true',
                        help='Use UFF module.')
    parser.add_argument('--use_ssl', default=False, action='store_true',
                        help='Use ssl module.')
    parser.add_argument('--save_preds', default=True, action='store_true',
                        help='Save predicts results.')
    parser.add_argument('--group_size', default=128, type=int,
                        help='Point group size of Point Transformer.')
    parser.add_argument('--num_group', default=1024, type=int,
                        help='Point groups number of Point Transformer.')
    parser.add_argument('--random_state', default=None, type=int,
                        help='random_state for random project')
    # parser.add_argument('--dataset_type', default='eyecandies', type=str, choices=['mvtec3d', 'eyecandies'],
    #                     help='Dataset type for training or testing')
    # parser.add_argument('--dataset_path', default='/home/c1/zgp/eyecandies/eyecandies_preprocessed', type=str,
    #                     help='Dataset store path')
    # parser.add_argument('--save_feature_path', default='/home/c1/zgp/datasets/patch_libEyeDFPFHDINO', type=str,
    #                     help='Save feature for training fusion block.')
    parser.add_argument('--dataset_type', default='mvtec3d', type=str, choices=['mvtec3d', 'eyecandies'],
                        help='Dataset type for training or testing')
    parser.add_argument('--dataset_path', default='/home/zgp/Documents/m3dmpre/datasets/mvtec3d', type=str,
                        help='Dataset store path')
    parser.add_argument('--save_feature_path', default='/home/c1/zgp/datasets/patch_libDFPFHDINO', type=str,
                        help='Save feature for training fusion block.')
    # parser.add_argument('--dataset_path', default='/home/code_project/zgp/Eyedataset/Eyecandies_preprocessed', type=str,
    #                     help='Dataset store path')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Images size for model')
    parser.add_argument('--xyz_s_lambda', default=1.0, type=float,
                        help='xyz_s_lambda')
    parser.add_argument('--xyz_smap_lambda', default=1.0, type=float,
                        help='xyz_smap_lambda')
    parser.add_argument('--rgb_s_lambda', default=0.1, type=float,
                        help='rgb_s_lambda')
    parser.add_argument('--rgb_smap_lambda', default=0.1, type=float,
                        help='rgb_smap_lambda')
    parser.add_argument('--fusion_s_lambda', default=1.0, type=float,
                        help='fusion_s_lambda')
    parser.add_argument('--fusion_smap_lambda', default=1.0, type=float,
                        help='fusion_smap_lambda')
    parser.add_argument('--coreset_eps', default=0.9, type=float,
                        help='eps for sparse project')
    parser.add_argument('--f_coreset', default=0.1, type=float,
                        help='eps for sparse project')
    parser.add_argument('--asy_memory_bank', default=None, type=int,
                        help='build an asymmetric memory bank for point clouds')
    parser.add_argument('--ocsvm_nu', default=0.5, type=float,
                        help='ocsvm nu')
    parser.add_argument('--ocsvm_maxiter', default=1000, type=int,
                        help='ocsvm maxiter')
    parser.add_argument('--rm_zero_for_project', default=False, action='store_true',
                        help='Save predicts results.')
    args = parser.parse_args()
    run_3d_ads(args)
