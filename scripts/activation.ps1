python main.py -s 1 -e 5 -p 0.0 --task activ
python main.py -s 1 -e 5 -p 0.1 --task activ
python main.py -s 1 -e 5 -p 0.2 --task activ
python main.py -s 1 -e 5 -p 0.4 --task activ
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 5 -p 0.0 --task activ
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 5 -p 0.1 --task activ
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 5 -p 0.2 --task activ
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.4 --task activ
python main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 1 -e 2 -p 0.0 --task activ
python main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 1 -e 2 -p 0.1 --task activ
python main.py -M ResNet18 -D CIFAR-10 -N 50000 -T 200 -s 1 -e 2 -p 0.2 --task activ