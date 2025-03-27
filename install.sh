#!/bin/bash

git config --global --add safe.directory /data/Github/chitu/third_party/AutoGPTQ
git config --global --add safe.directory /data/Github/chitu/third_party/cutlass
git config --global --add safe.directory /data/Github/chitu/third_party/EETQ
git config --global --add safe.directory /data/Github/chitu/third_party/FlashMLA
git config --global --add safe.directory /data/Github/chitu/third_party/llm-awq
git config --global --add safe.directory /data/Github/chitu/third_party/muxi_layout_kernels
git config --global --add safe.directory /data/Github/chitu/third_party/muxi_w8a8_kernels
git config --global --add safe.directory /data/Github/chitu/third_party/nv_w8a8_kernels
git config --global --add safe.directory /data/Github/chitu/third_party/spdlog

#pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple

#git clone --recursive https://github.com/thu-pacman/chitu && cd chitu

sudo apt install rustc -y
sudo apt install cargo -y

#curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#source $HOME/.cargo/env
#pip install --upgrade pip
#pip install tiktoken

pip install flashinfer-python

export CUDA_HOME=/usr/local/cuda
#export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/miniconda/envs/chitu/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

pip install -r requirements-build.txt
#pip install -U torch --index-url https://download.pytorch.org/whl/cu126
pip install -U torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --extra-index-url https://download.pytorch.org/whl/cu126
TORCH_CUDA_ARCH_LIST=7.5 CHITU_SETUP_JOBS=40 MAX_JOBS=40 pip install --no-build-isolation .