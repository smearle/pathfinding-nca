# Using python 3.9 w/ GPU (pyg a.k.a. torch_geometric does not support python 3.10 w/ GPU), otherwise 3.10
python -m pip install -r requirements.txt


## Install pytorch using conda
# TODO: check for OS to detemine which of below commands to use.

# For CPU-only:
# conda install pytorch torchvision torchaudio -c pytorch

# For most GPUs:
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# For 3090:
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# No distinction between CUDA/CPU
conda install pyg -c pyg


# Recommended speedup. Probably not important for us.
conda install scikit-learn-intelex