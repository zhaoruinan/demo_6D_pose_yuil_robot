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
