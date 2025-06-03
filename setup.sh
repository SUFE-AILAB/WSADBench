conda create -n ad python=3.9 -y
conda activate ad

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install tensorflow
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install tf-nightly==2.20.0.dev20250327
pip install -r requirements.txt
pip install pytorchvideo==0.1.5
# pip install git+https://github.com/facebookresearch/pytorchvideo.git
pip install opencv-python==4.11.0.86
# -i https://pypi.tuna.tsinghua.edu.cn/simple
