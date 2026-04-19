now_path=`pwd`

# Install miniconda
cd
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# Configure ~/.bashrc
echo alias p=\"ps -aux|grep zhangjq|grep 'python -u'\" >> ~/.bashrc
echo alias n=\'nvidia-smi\' >> ~/.bashrc
echo alias d=\'du -hs * | sort -h\' >> ~/.bashrc
echo alias del_pycache=\'find . -type d -name __pycache__ -prune -exec rm -rf {} \;\' >> ~/.bashrc

echo export PIP_CACHE_DIR='$PWD'/tmp >> ~/.bashrc
echo export TMPDIR='$PWD'/tmp >> ~/.bashrc

# Install python packages
cd $now_path
source ~/.bashrc
conda env create -f env_cuda_latest.yaml
python generate_Cifar10.py noniid - dir
python generate_Cifar100.py noniid - dir
python generate_TinyImagenet.py noniid - dir

改变异构程度的地方在 dataset\utils下的 dataset_utils.py文件

python generate_Cifar10.py noniid - pat
python generate_Cifar100.py noniid - pat
python generate_TinyImagenet.py noniid - pat

改变异构程度的地方在各个generate_xx.py文件中