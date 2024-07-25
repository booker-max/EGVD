apt update && apt install -y libsm6 libxext6 libxrender-dev
pip install pandas tensorboardX scikit-image tensorboard commentjson h5py matplotlib einops numpy==1.22.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
# python /code/RMFD_bita/train_mprnet/pytorch-gradual-warmup-lr/setup.py install
pip install lpips einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install commentjson -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip uninstall numpy -y
pip install --upgrade numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install fvcore -i https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p /root/.cache/torch/hub/checkpoints/
cp /data/booker/LN_base/pretrain_dict/VDR-STF/vgg19-dcbb9e9d.pth /root/.cache/torch/hub/checkpoints/
cp /data/booker/LN_base/pretrain_dict/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/
cp /data/booker/LN_base/pretrain_dict/resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/
cp /code/SupervisedDR/eavd/baseline/usdrain/brisque_svm_weights.pth /root/.cache/torch/hub/checkpoints/
cp /data/booker/LN_base/Code_redrain/USDerain_v2/train_lnr/networks/basic/IQA-PyTorch/mat_file/niqe_modelparameters.mat /root/.cache/torch/hub/checkpoints/
cp /data/booker/LN_base/Code_redrain/USDerain_v2/train_lnr/networks/basic/IQA-PyTorch/mat_file/NRQM_model.mat /root/.cache/torch/hub/checkpoints/

# cp /data/booker/LN_base/pretrain_dict/VDR-STF/vgg19-dcbb9e9d.pth /root/.cache/torch/hub/checkpoints/
apt-get install zip -y
# pip install scikit-image -i https://pypi.mirrors.ustc.edu.cn/simple
# sh /code/SupervisedDR/networks/baseline/savd/modules/DCNv2/make.sh
pip install numpy==1.15.0
pip install --upgrade numpy
pip install torchsummary


# pip install pyiqa -i https://pypi.mirrors.ustc.edu.cn/simple
# pip3 install --upgrade protobuf==3.20.1
# pip uninstall timm -y
# pip install timm -i https://pypi.tuna.tsinghua.edu.cn/simple

# pip install guided_filter_pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install cupy-cuda111 -i https://pypi.tuna.tsinghua.edu.cn/simple

###for torch0.4
# pip install guided_filter_pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install cupy-cuda111 -i https://pypi.tuna.tsinghua.edu.cn/simple
# sh /code/RMFD_bita/train_lnr/networks/SAVD/modules/DCNv2/make.sh
# cp /data/booker/LN_base/network-default.pytorch /root/.cache/torch/hub/checkpoints/pwc-default