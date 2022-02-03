# -*- coding: utf-8 -*-
import cdt
import bnlearn as bn
import torch
import pandas as pd
import pickle
import os

from cdt.causality.graph import SAM, CAM, CCDr, GES, IAMB, PC, LiNGAM, MMPC, GS
from cdt.causality.graph import CGNN

import networkx as nx
import matplotlib.pyplot as plt
from cdt.data import load_dataset
import numpy as np
from causalnex.structure.notears import from_pandas
from causalnex.structure import StructureModel

from configparser import ConfigParser
config_file = "config.ini"
parser = ConfigParser()
parser.optionxform = str
parser.read(config_file)

def set_config(section, option, value):
    if parser.has_section(section):
        parser.set(section, option , value)
        with open(config_file, "w") as conf_file:
            parser.write(conf_file)
            return True

# datass = [ "asia", "sachs", "alarm", "water", "andes" ] 
# datass = [ "asia", "sachs", "alarm", "andes" "child", "coronary", "insurance"] 
datass = [ 'andes', 'diabetes', 'alarm'] 

# algo_names = ['ccdr', 'ges', 'gs', 'iamb', 'lingam', 'mmpc', 'pc', 'notears']
# algo_names = [ 'pc', cgnn]
# algo_names = ['cam', 'ccdr', 'ges', 'gs', 'iamb', 'lingam', 'mmpc', 'pc']
algo_names = ['lingam', 'notears', 'ccdr', 'sam']

def get_noTears(d, dropped, d_names=''):
    data = pd.read_csv(f"Datasets/{d}.csv")
    sm = from_pandas(data)
    df = nx.to_pandas_adjacency(sm, nodelist=list(data.columns))
    if dropped:
        df.to_csv(f"adj_mats/dropped/{d}_notears{d_names}.csv")
    else:
        df.to_csv(f"adj_mats/{d}_notears.csv")
    return sm

def cdt_to_df(obj):
    dict(obj.adj)
    adj_dict = {}
    for n1 in obj.adj.keys():
        adj_dict[n1] = {}
        for n2 in obj.adj.keys():
            adj_dict[n1][n2] = 0
        for n2 in obj.adj[n1]:
            adj_dict[n1][n2] = obj.adj[n1][n2]['weight']

    adj_dict        
    df = pd.DataFrame.from_dict(adj_dict)
    return df

def run_algos(datass, algo_names, dropped=False, drop_indices=[]):
    allowed = ['cam', 'ccdr', 'ges', 'gs', 'iamb', 'lingam', 'mmpc', 'pc', 'notears', 'dagGNN']
    for data_name in datass:

        # data = bn.import_example(data_name)
        data = pd.read_csv(f"Datasets/{data_name}.csv")
        d_names = ''
        if dropped:
            
            for j in drop_indices: d_names = (d_names+'_'+str(j)) 

            for i in drop_indices:
                data = data.drop(data.columns[i], axis=1)

        for algo_name in algo_names :
            print(data_name, algo_name)
            if algo_name == 'notears':
                get_noTears(data_name, dropped, d_names)

            elif algo_name == 'dagGNN':
                data.to_csv('algorithms/DAG_from_GNN/datasets/dropped.csv')
                # set_config('Device','data_filename','')
                os.chdir("algorithms/DAG_from_GNN")
                print(os.system("pwd"))
                os.system(f'python -m DAG_from_GNN')
                os.chdir("../..")
                print(os.system("pwd"))
                if dropped:
                    os.system(f'mv algorithms/DAG_from_GNN/results/final_adjacency_matrix.csv adj_mats/dropped/{data_name}_{algo_name}{d_names}.csv')
                    os.system(f'mv algorithms/DAG_from_GNN/results/DAG_plot.png cdt_graphs/dropped/{data_name}_{algo_name}{d_names}.csv')
                else:
                    os.system(f'mv algorithms/DAG_from_GNN/results/final_adjacency_matrix.csv adj_mats/{data_name}_{algo_name}.csv')
                    os.system(f'mv algorithms/DAG_from_GNN/results/DAG_plot.png cdt_graphs/{data_name}_{algo_name}.csv')
                print('dagGNN')

            elif algo_name in allowed :
                try:
                    if algo_name=='sam':
                        obj = SAM()
                    elif algo_name =='cam':
                        obj = CAM()
                    elif algo_name =='ccdr':
                        obj = CCDr()
                    elif algo_name == 'ges':
                        obj = GES()
                    elif algo_name == 'gs':
                        obj = GS()
                    elif algo_name == 'iamb':
                        obj = IAMB()
                    elif algo_name == 'pc':
                        obj = PC()
                    elif algo_name == 'lingam':
                        obj = LiNGAM()
                    elif algo_name == 'mmpc':
                        obj = MMPC()
                    elif algo_name == 'cgnn':
                        obj = CGNN()

                    print("wohoo")
                    # output = obj.create_graph_from_data(data)
                    output = obj.predict(data)    #No graph provided as an argument
                    adj_m = (output.adj)
                    print('y')
                    nx.draw_networkx(output, font_size=8)
                    print('ok')
                    print(adj_m)
                    if dropped :
                        
                        pickle.dump(output, open(f"cdt_dags/dropped/{data_name}_{algo_name}{d_names}.pkl", 'wb'))
                        plt.savefig(f"cdt_graphs/dropped/{data_name}_{algo_name}{d_names}.png", format="PNG")

                    else:
                        pickle.dump(output, open(f"cdt_dags/{data_name}_{algo_name}.pkl", 'wb'))
                        plt.savefig(f"cdt_graphs/{data_name}_{algo_name}.png", format="PNG")

                    plt.show()

                    df = cdt_to_df(output)
                    if dropped:
                        df.to_csv(f"adj_mats/dropped/{data_name}_{algo_name}{d_names}.csv")
                    else:
                        df.to_csv(f"adj_mats/{data_name}_{algo_name}.csv")
                    # i=0
                    # var_mapping = {}
                    # print(adj_m)
                    # for var in list(adj_m.keys()):
                    #     var_mapping[var] = i
                    #     i+=1
                    # shape = (len(list(var_mapping.keys())),len(list(var_mapping.keys())))
                    # adj_np = np.zeros(shape)
                    # for i in adj_m:
                    #     for j in adj_m[i]:
                    #         x, y = var_mapping[i], var_mapping[j]
                    #         adj_np[x][y] = adj_m[i][j]['weight']
                    # print(adj_np)
                    # print(adj_m)
                    # if dropped :
                    #     np.savetxt(f"adj_mats/dropped_{data_name}_{algo_name}.csv", adj_np, delimiter = ",")
                    # else :
                    #     np.savetxt(f"adj_mats/{data_name}_{algo_name}.csv", adj_np, delimiter = ",")
                except:
                    print("error:", algo_name, data_name )

        # print("wohoo!!!!!")

        # for 


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    run_algos(datass, algo_names)
    # run_algos(['asia', 'sachs', 'andes'], ['dagGNN'])



