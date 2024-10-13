export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=true
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_DIR=$SCRIPT_DIR

env_name=sam2
INSTALLED_GCC_VERSION=9.5.0
INSTALLED_CUDA_VERSION=12.4.1
INSTALLED_CUDA_ABBREV=cu124

conda create --name $env_name --yes python=3.10

eval "$(conda shell.bash hook)"
conda activate ${env_name}

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

conda install -y -c conda-forge sysroot_linux-64=2.17 ffmpeg gxx=${INSTALLED_GCC_VERSION}
conda install -y -c "nvidia/label/cuda-${INSTALLED_CUDA_VERSION}" cuda

echo ${conda_home}

which python
which pip
which nvcc

export BUILD_WITH_CUDA=1
export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export FORCE_CUDA=1
export MAX_JOBS=12
export AM_I_DOCKER=False
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# install sam2
cd $REPO_DIR/third_party/sam2
pip install -e .

# also install sam1
pip install git+https://github.com/facebookresearch/segment-anything.git

pip install ipykernel ffmpeg-python imageio-ffmpeg
