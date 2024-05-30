from itertools import combinations
import numpy as np
import networkx as nx
import time
from typing import List, Tuple
from tqdm import trange
import torch
import os

from param import param
from toric_code import ToricCode
from train import NeuralNetwork

SIZE = 5

'''
↓シミュレーションとその評価
'''
def evaluate(n_iter):
    count = 0
    toric_code = ToricCode()
    current_directory = os.getcwd()
    model_directory_path = os.path.join(current_directory, 'learned_model')
    os.chdir(model_directory_path)
    model = NeuralNetwork()
    model.load_state_dict(torch.load('NN_'+str(SIZE)+'.pt'))

    spent_decode = 0
    count_no_error = 0
    for _ in trange(n_iter):
        errors = toric_code.generate_errors()
        '''print('errors') #for debug
        print(errors) #for debug'''
        is_no_error = np.all(errors == 0)
        if is_no_error:
            count_no_error += 1
            continue
        #syndrome_x = toric_code.generate_syndrome_X(errors)
        '''print('syndrome_x') #for debug
        print(syndrome_x) #for debug'''
        #syndrome_z = toric_code.generate_syndrome_Z(errors)
        '''print('syndrome_z') #for debug
        print(syndrome_z) #for debug'''
        syn_x = torch.from_numpy(toric_code.generate_syndrome_X(errors))
        syn_z = torch.from_numpy(toric_code.generate_syndrome_Z(errors))
        syn = torch.cat((syn_x, syn_z), dim=0)
        syn = syn.float()
        before = time.perf_counter()
        pred = model(syn.flatten())
        pred = pred.view((4*SIZE, SIZE))
        chunks = torch.chunk(pred, 2, dim=0)
        x_map, z_map = chunks
        #print(x_map)
        #print(z_map)
        x_map = (x_map >= 0.5).int()
        z_map = (z_map >= 0.5).int()
        x_map = x_map.numpy()
        z_map = z_map.numpy()
        #print(x_map)
        #print(z_map)
        errors = toric_code.operate_x(errors, z_map)
        errors = toric_code.operate_z(errors, x_map)
        #print(errors)
        spent_decode += time.perf_counter() - before
        if np.all(toric_code.generate_syndrome_X(errors)==0) and np.all(toric_code.generate_syndrome_Z(errors)==0): 
            if toric_code.not_has_non_trivial_x(errors) and toric_code.not_has_non_trivial_z(errors):
                count = count + 1
    print('decode         : ' + str(spent_decode / n_iter) + str(" seconds"))
    print(f"logical error rates: {n_iter - count_no_error - count}/{n_iter - count_no_error}", (n_iter - count_no_error - count) / (n_iter - count_no_error))
#実行
print("Toric code simulation // code distance is " + str(param.code_distance))
evaluate(10000)