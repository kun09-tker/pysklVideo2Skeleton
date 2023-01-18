echo "install dependencies: (use cu111 because colab has CUDA 11.1)"
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

echo "install mmcv-full thus we could use CUDA operators"
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

echo "Download and install mmdetection"
# Install mmdetection
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

pip install -e .

# Install some optional requirements
pip install -r requirements/optional.txt

echo "Download and install mmpose"
cd ..
rm -rf mmpose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
# Install some optional requirements
pip install -r requirements/optional.txt
cd ..

echo "install scipy"
pip install scipy

echo "Update numpy"
pip uninstall numpy -y
pip install numpy

echo "Install torch"
pip install torch>=1.5

echo "Install moviepy"
pip install moviepy

echo "Install imageio"
pip3 install imageio==2.4.1