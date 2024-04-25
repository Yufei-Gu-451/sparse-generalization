#$ -S /bin/bash
#$ -j y
#$ -l tmem=16G
#$ -l h_rt=200:00:00
#$ -l gpu=true
#$ -N SLT-2
source /share/apps/source_files/python/python-3.8.5.source
date
python3 sparse-generalization/separate_layer_test_2.py
date