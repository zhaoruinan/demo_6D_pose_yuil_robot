# demo_6D_pose_yuil_robot

![introduction](./assets/introduction.png)
```
# Prepare for demo code


# Download demo code and run.
git clone https://github.com/zhaoruinan/demo_6D_pose_yuil_robot.git

docker pull ruinanzhao/pvnet-clean:latest

sudo vim ~/.bashrc

# add this lines,use "i" to edit
export PVNET_DOCKER=ruinanzhao/pvnet-clean:latest
export PVNET_GIT=$HOME/demo_6D_pose_yuil_robot  # update this with the directory of demo code
source $PVNET_GIT/docker/setup_dev.bash

# use ":wq" to save and exit
source ~/.bashrc

pvnet_docker # By this, a docker env will be run for demo code.

# inside the docker container
conda activate pvnet
python run.py --type online2 --cfg_file configs/custom.yaml

```

# rebuild docker image
```
cd docker
docker build . -t  ruinanzhao/pvnet-clean:latest
```

# 

