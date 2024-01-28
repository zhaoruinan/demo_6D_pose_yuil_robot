# yuil_robot_python

![introduction](./assets/introduction.png)
Camera intrinsic parameters matrix:
[[812.41703757   0.         738.73620289]
 [  0.         810.31869773 339.889511  ]
 [  0.           0.           1.        ]]

Camera distortion coefficients:
[[ 0.04252043  0.05231824 -0.00314384 -0.00499403 -0.18527505]]

www.autodl.com

PyTorch  1.11.0
Python  3.8(ubuntu20.04)
Cuda  11.3

wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh &&     sh ./Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -p /opt/conda &&     rm ./Miniconda3-py37_4.8.3-Linux-x86_64.sh &&     export PATH=$PATH:/opt/conda/bin &&     conda install conda-build
conda init bash
exit

git clone https://gitlab.ruinan.top/zhaoruinan/pv_net_docker.git 
cd pv_net_docker/
apt install -yq         nano         sudo         wget         curl         build-essential         cmake         git         ca-certificates         python3.7         python3-pip         libjpeg-dev         zip         unzip         libpng-dev         libeigen3-dev         libglfw3-dev         libglfw3         libgoogle-glog-dev         libsuitesparse-dev         libatlas-base-dev
apt update
apt install libglfw3-dev
apt install libglfw3
apt install gcc-7 g++-7 libsm6 libxrender1 libfontconfig1 -y
apt install -yq         nano         sudo         wget         curl         build-essential         cmake         git         ca-certificates         python3.7         python3-pip         libjpeg-dev         zip         unzip         libpng-dev         libeigen3-dev         libglfw3-dev         libglfw3         libgoogle-glog-dev         libsuitesparse-dev         libatlas-base-dev
apt install libjpeg-dev         zip         unzip         libpng-dev         libeigen3-dev         libglfw3-dev         libglfw3         libgoogle-glog-dev         libsuitesparse-dev         libatlas-base-dev
pip install -r requirements.txt
pip install --user yacs==0.1.4 numpy==1.18.0 opencv-python tqdm==4.28.1 pycocotools==2.0.0 matplotlib==2.2.2 
pip install --user Cython==0.28.2 yacs open3d-python==0.3.0.0 opencv-python pycocotools plyfile 
conda install -c open3d-admin open3d
pip install skimage
pip install scikit-image
cd  lib/csrc/ransac_voting/
export CUDA_HOME="/usr/local/cuda-11.3"
python setup.py build_ext --inplace
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
python setup.py build_ext --inplace
cd ../nn
python setup.py build_ext --inplace
cd ../fps
python setup.py build_ext --inplace
cd ../..

python train_net.py --cfg_file configs/linemod.yaml model mycat cls_type cat


nvcc -V
ln -s /root/autodl-tmp/LINEMOD linemod
watch -n 1 nvidia-smi 
docker system prune -a


conda activate oneposepp
   35  cd  /opt/conda/envs/oneposepp/
   36  ls
   37  cd lib/python3.7/
   38  ls
   39  git clone https://github.com/hjwdzh/DeepLM.git
   40  cd DeepLM/
   41  ./example.sh