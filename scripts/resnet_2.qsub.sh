#$ -S /bin/bash
#$ -j y
#$ -l tmem=16G
#$ -l h_rt=100:00:00
#$ -l gpu=true
#$ -N test_run
source /share/apps/source_files/python/python-3.8.5.source
date
python3 sparse-generalization/main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 3 -e 5 -p 0.1 --task init
date
python3 sparse-generalization/main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 3 -e 5 -p 0.1 --task train
date