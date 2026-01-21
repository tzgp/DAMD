# Density-Augmented Adaptive Feature Representation for Multimodal Detection of Industrial Subtle Defects (Under review at The Visual Computer)
If you find this work useful, please cite the manuscript upon publication.
## Setup
We implement this repo with the following environment:
- Python 3.8
Install the other package via:
``` bash
pip install -r requirement.txt
# install knn_cuda
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# install pointnet2_ops_lib
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
## Data Download and Preprocess
### Dataset
- The `MVTec-3D AD` dataset can be download from the [Official Website of MVTec-3D AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad). 
- The `Eyecandies` dataset can be download from the [Official Website of Eyecandies](https://eyecan-ai.github.io/eyecandies/). 
After download, put the dataset in `dataset` folder.
### Datapreprocess
To run the preprocessing (As M3DM)
```bash
python utils/preprocess_eyecandies.py
python utils/preprocessing.py datasets/mvtec3d/
```
## Checkpoints
You can download ViT-b/8 at [M3DM](https://github.com/nomewang/M3DM)
## Run our method 
```bash
nohup stdbuf -oL -eL python3.8 main.py --method_name DINO+FPFH+add+late > 3080DINODFPFHaddenlog.txt 2>&1 & 
```
(This framework also includes methods for our failed attempts, which can be ignored.)
## Thanks
Our repo is built on [3D-ADS](https://github.com/eliahuhorwitz/3D-ADS) and [M3DM](https://github.com/nomewang/M3DM), thanks their extraordinary works!
 


