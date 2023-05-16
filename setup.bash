apt update
apt install python3-pip nano 
apt-get update
apt-get install git ffmpeg libsm6 libxext6  -y
pip3 install torch torchvision torchaudio ftfy regex tqdm timm opencv-python gensim nltk scikit-learn imageio fvcore matplotlib seaborn igraph grad-cam
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/epfml/sent2vec.git


# export PYTHONPATH="${PYTHONPATH}:own_zsar"