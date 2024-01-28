# README

## Build 

```bash
docker build -t pvnet_clean:latest .
```

## Run

To run the docker
Add the following to your ~/.bashrc

```bash
export PVNET_DOCKER=pvnet_clean:latest
export PVNET_GIT=$HOME/gits/clean-pvnet  # update
source $PVNET_GIT/docker/setup_dev.bash
```

run it with:

```bash
pvnet_docker
```
cd pvnet
conda activate pvnet
pip3  install --user   Pillow==6.2.1 imagezmq
python run.py --type online --cfg_file configs/custom.yaml
