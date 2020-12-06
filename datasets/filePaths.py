import os

TRAINS_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], './datas/trains.txt')
FB15K_BASE_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], './datas/FB15K')
FB15K_BASE_PATH_1_1_PATH = os.path.join(FB15K_BASE_PATH, '1-1.txt')
FB15K_BASE_PATH_1_n_PATH = os.path.join(FB15K_BASE_PATH, '1-n.txt')
FB15K_BASE_PATH_n_1_PATH = os.path.join(FB15K_BASE_PATH, 'n-1.txt')
FB15K_BASE_PATH_n_n_PATH = os.path.join(FB15K_BASE_PATH, 'n-n.txt')

