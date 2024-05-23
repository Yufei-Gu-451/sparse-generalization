#$ -S /bin/bash
#$ -j y
#$ -l tmem=16G
#$ -l h_rt=100:00:00
#$ -l gpu=true
#$ -N ViT
source /share/apps/source_files/python/python-3.8.5.source
date
python3 sparse-generalization/main.py -M ViT -D MNIST -N 50000 -T 1000 -p 0.0 -s 1 -e 1 --opt adam --task init
python3 sparse-generalization/main.py -M ViT -D MNIST -N 50000 -T 1000 -p 0.0 -s 1 -e 1 --opt adam --task train
python3 sparse-generalization/main.py -M ViT -D CIFAR-10 -N 50000 -T 1000 -p 0.0 -s 1 -e 1 --opt adam --task init
python3 sparse-generalization/main.py -M ViT -D CIFAR-10 -N 50000 -T 1000 -p 0.0 -s 1 -e 1 --opt adam --task train
date