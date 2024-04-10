#$ -S /bin/bash
#$ -j y
#$ -l tmem=16G
#$ -l h_rt=100:00:00
#$ -l gpu=true
#$ -N test_run
source /share/apps/source_files/python/python-3.8.5.source
date
python main.py -s 1 -e 3 -p 0.6 --task init
python main.py -s 1 -e 3 -p 0.6 --task train
python3 sparse-generalization/main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.6 --task init
python3 sparse-generalization/main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.6 --task train
date