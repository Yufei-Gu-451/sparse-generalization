#$ -S /bin/bash
#$ -j y
#$ -l tmem=16G
#$ -l h_rt=100:00:00
#$ -l gpu=true
#$ -N test_run
source /share/apps/source_files/python/python-3.8.5.source
date
python3 sparse-generalization/main.py -s 997 -e 998 -p 0.0 --task init
date
python3 sparse-generalization/main.py -s 997 -e 998 -p 0.0 --task train
date