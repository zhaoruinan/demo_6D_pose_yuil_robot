# README

## Build 

```bash
docker build -t ruinanzhao/pvnet_clean:latest .
```

## Run

To run the docker
Add the following to your ~/.bashrc

```bash
export PVNET_DOCKER=ruinanzhao/pvnet_clean:latest
export PVNET_GIT=$HOME/demo_6D_pose_yuil_robot  # update
source $PVNET_GIT/docker/setup_dev.bash
```

run it with:

```bash
pvnet_docker
```
cd demo_6D_pose_yuil_robot
conda activate pvnet
python run.py --type online2 --cfg_file configs/custom.yaml



  1  conda activate pvnet
    2  python run.py --type demo
    3  cd demo_6D_pose_yuil_robot/
    4  python run.py --type demo
    5  ls
    6  cd lib
    7  ls
    8  cd networks/
    9  ls
   10  cd ..
   11  python run.py --type online2
   12  pip install gitpython
   13* python run.py --type online2 
   14  python run.py --type demo --cfg_file configs/custom.yaml
   15  python run.py --type online2 --cfg_file configs/custom.yaml
   16  python run.py --type online --cfg_file configs/custom.yaml
   17  python run.py --type online2 --cfg_file configs/custom.yaml
   18  python run.py --type detector --cfg_file configs/custom.yaml
   19  python run.py --type detector_pvnet --cfg_file configs/custom.yaml
   20  python run.py --type online2 --cfg_file configs/custom.yaml
   21  python run.py --type demo --cfg_file configs/custom.yaml
   22  python run.py --type detector_pvnet --cfg_file configs/custom.yaml
   23  python run.py --type online3 --cfg_file configs/custom.yaml
   24  python run.py --type detector_pvnet --cfg_file configs/custom.yaml
   25  python run.py --type online3 --cfg_file configs/custom.yaml
   26  python run.py --type demo --cfg_file configs/custom.yaml
   27  python
   28  python run.py --type demo --cfg_file configs/custom.yaml
   29  python run.py --type demo3 --cfg_file configs/custom.yaml
   30  python run.py --type online3 --cfg_file configs/custom.yaml
   31  python run.py --type visualize --cfg_file configs/custom.yaml
   32  python run.py --type online3 --cfg_file configs/custom.yaml
   33  python run.py --type demo3 --cfg_file configs/custom.yaml
   34  python run.py --type online3 --cfg_file configs/custom.yaml
   35  python run.py --type demo3 --cfg_file configs/custom.yaml
   36  python
   37  python kpt3d.py 
   38  python run.py --type demo3 --cfg_file configs/custom.yaml
   39  python run.py --type online3 --cfg_file configs/custom.yaml
   40  python run.py --type demo3 --cfg_file configs/custom.yaml
   41  python run.py --type online3 --cfg_file configs/custom.yaml
   42  python run.py --type demo3 --cfg_file configs/custom.yaml
   43  python run.py --type online3 --cfg_file configs/custom.yaml
   44  history
(pvnet) root@pvnet_dev:/home/zhuzhuxia/demo_6D_pose_yuil_robot# 
005
