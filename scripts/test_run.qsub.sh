#$ -S /bin/bash
#$ -l tmem=16G
#$ -l h_rt=1:00:00
#$ -l gpu=true
#$ -N test_run
date
python3 main.py -s 999 -e 999 -p 0.0 -task init
date
python3 main.py -s 999 -e 999 -p 0.0 -task train
date