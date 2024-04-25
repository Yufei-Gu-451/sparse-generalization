#$ -S /bin/bash
#$ -j y
#$ -l tmem=16G
#$ -l h_rt=150:00:00
#$ -l gpu=true
#$ -N ResNet18
source /share/apps/source_files/python/python-3.8.5.source
date
python3 sparse-generalization/main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 1 -e 2 -p 0.0 --task test --knn True
python3 sparse-generalization/main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 1 -e 2 -p 0.1 --task test --knn True
python3 sparse-generalization/main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 1 -e 2 -p 0.2 --task test --knn True
date55