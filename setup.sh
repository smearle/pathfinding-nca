## Install pytorch using conda
# TODO: check for OS to detemine which of below commands to use.

# For CPU-only:
# conda install pytorch torchvision torchaudio -c pytorch

# For most GPUs:
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c nvidia
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# For 3090:
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# No distinction between CUDA/CPU
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
# conda install pyg -c pyg


# Recommended speedup. Probably not important for us.
conda install scikit-learn-intelex



# Using python 3.10
python -m pip install -r requirements.txt
