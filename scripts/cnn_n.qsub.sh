#$ -S /bin/bash
#$ -j y
#$ -l tmem=16G
#$ -l h_rt=100:00:00
#$ -l gpu=true
#$ -N test_run
source /share/apps/source_files/python/python-3.8.5.source
date
python3 sparse-generalization/main.py -d CIFAR-10 -N 50000 -m CNN --epochs 200 -s 2 -e 3 -p 0.4 --task init
date
python3 sparse-generalization/main.py -d CIFAR-10 -N 50000 -m CNN --epochs 200 -s 2 -e 3 -p 0.4 --task train
date