#python main.py -s 1 -e 3 -p 0.6 --task test
#python main.py -s 1 -e 3 -p 0.6 --task test --knn True
#python main.py -s 1 -e 3 -p 0.6 --task test --rade True
#python main.py -s 1 -e 3 -p 0.6 --task activ
#python main.py -s 1 -e 3 -p 0.6 --task sparse

python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.6 --task test
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.6 --task test --knn True
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.6 --task test --rade True
python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.6 --task activ
#python main.py -M CNN -D CIFAR-10 -N 50000 -T 200 -s 1 -e 3 -p 0.6 --task sparse
