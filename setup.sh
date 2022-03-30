python -m pip install -r requirements.txt

# for writing video
conda install -c conda-forge ffmpeg

# for torch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# for 3090
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch