#$ -S /bin/bash
#$ -j y
#$ -l tmem=16G
#$ -l h_rt=100:00:00
#$ -l gpu=true
#$ -N test_run
source /share/apps/source_files/python/python-3.8.5.source
date
python3 sparse-generalization/main.py -d CIFAR-10 -N 50000 -m ResNet18 --epochs 200 -s 4 -e 5 -p 0.2 --task init
date
python3 sparse-generalization/main.py -d CIFAR-10 -N 50000 -m ResNet18 --epochs 200 -s 4 -e 5 -p 0.2 --task train
date