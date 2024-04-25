python main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 3 -e 5 -p 0.0 --task test
python main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 3 -e 5 -p 0.1 --task test
python main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 3 -e 5 -p 0.2 --task test
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 5 -p 0.0 --task test
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 5 -p 0.1 --task test
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 5 -p 0.2 --task test
python main.py -s 1 -e 5 -p 0.0 --task test
python main.py -s 1 -e 5 -p 0.1 --task test
python main.py -s 1 -e 5 -p 0.2 --task test
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.4 --task test
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.6 --task test
python main.py -s 1 -e 3 -p 0.4 --task test
python main.py -s 1 -e 3 -p 0.6 --task test

python main.py -s 1 -e 5 -p 0.0 --task scale
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 5 -p 0.0 --task scale
python main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 3 -e 5 -p 0.0 --task scale