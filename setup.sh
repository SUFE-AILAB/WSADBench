conda create -n ad python=3.9 -y
conda activate ad

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install tensorflow
pip3 install torch torchvision torchaudio
pip install tf-nightly
pip install -r requirements.txt
pip install pytorchvideo
# pip install git+https://github.com/facebookresearch/pytorchvideo.git
pip install opencv-python
# -i https://pypi.tuna.tsinghua.edu.cn/simple