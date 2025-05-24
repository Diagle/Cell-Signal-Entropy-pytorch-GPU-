import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import seaborn as sns
import igraph as ig
import sys
sys.path.append('/home/kazundo/WorkSpace/')
from MyModule import MyPreprocession
from typing import Optional,Union,Any,Set,List,Tuple,Dict
import torch

def caculate_Prob_inv(x:torch.tensor,net:torch.tensor)->torch.tensor:
    x1 = x.reshape(-1, 1)
    Sum = x @ net @ x1
    if Sum == 0:
        return np.zeros_like(x)
    return torch.tensor(net @ x1 * x1).reshape((1,-1)) /Sum

def caculate_Prob_exp(x:torch.tensor,net:torch.tensor)->torch.tensor:
    temp = net*x
    pij_sum = torch.sum(temp,axis=1)
    pij_sum = torch.where(pij_sum == 0, torch.tensor(1.0, device=pij_sum.device), pij_sum)
    result = temp/pij_sum.reshape((-1,1))
    # result[result==0]=1
    return result


def caculate_MaxSr(adjust_matrix:pd.DataFrame,device='GPU')->float:
    with torch.no_grad():
        net = adjust_matrix.to_numpy(dtype=float)
        if device == 'GPU' and torch.cuda.is_available():
            net = torch.from_numpy(net).cuda()
    
        eigenvalue, featurevector = torch.linalg.eig(net)
        Max_Sr = torch.log2(torch.max(torch.real(eigenvalue)))
        return Max_Sr.cpu().item()
    
def caculate_entroph_sum(p:torch.tensor,pi:torch.tensor,net:torch.tensor,is_normal:bool=False)->np.array:
    p = torch.where(p == 0, torch.tensor(1.0, device=p.device), p)
    if  is_normal:
        log_ki = torch.log2(torch.sum(net,dim=1))
        log_ki[log_ki==0]=1
        Si = -1*torch.sum(torch.log2(p)*p,dim=1)/log_ki
    else:
        Si = -1*torch.sum(torch.log2(p)*p,dim=1)
    Sr = Si @ pi.reshape((-1,1))
    return Sr.cpu().item()

def get_cell_entroph(df:pd.DataFrame,adjust_matrix:pd.DataFrame,device='GPU')->float:
    if device == 'GPU' and torch.cuda.is_available():
        x = torch.tensor(df.values, dtype=torch.float, device='cuda')
        net = torch.tensor(adjust_matrix.values, dtype=torch.float, device='cuda')
    else:
        x = torch.tensor(df.values, dtype=torch.float, )
        net = torch.tensor(adjust_matrix.values, dtype=torch.float)
    weight = caculate_Prob_exp(x,net)
    print('pij finished')
    PI = caculate_Prob_inv(x,net)
    print('pi finished')
    Sr = caculate_entroph_sum(weight,PI,net)
    print('entropy finished')
    del net, x
    torch.cuda.empty_cache()
    return Sr

def pij_Correlation(x):
    return 0

def pij_RelationIndex(x):
    return 0


def extract_max_sub_nodes(net:pd.DataFrame)->List:
    import igraph as ig
    g = ig.Graph.Adjacency(np.array(net),mode="undirected")
    print(ig.summary(g))

    g.vs['genes'] = list(net.columns)
    select_v = g.connected_components().giant().vs['genes']
    print(len(select_v))
    return select_v

def get_Sr(df,net,method=None):
    df1 = MyPreprocession.normal_hvgs(df,method,0)
    max_sub_nodes = extract_max_sub_nodes(net)
    max_sub_net= net.loc[max_sub_nodes,max_sub_nodes]
    df2 =  df1.loc[:,max_sub_nodes]
    Max_Sr = caculate_MaxSr(max_sub_net)
    result = df2.apply(lambda x:get_cell_entroph(x,max_sub_net)/Max_Sr,axis=1)
    return result




if __name__ == '__main__':
    np.random.seed(114514)
    test_df = pd.DataFrame(np.random.randint(0,20,(100,5)))
    test_net = pd.DataFrame(np.array([
        [0,1,1,1,0],
        [1,0,1,1,0],
        [1,1,0,1,0],
        [1,1,1,0,0],
        [0,0,0,0,0]
         ]
    ))

    max_sub_nodes = extract_max_sub_nodes(test_net)
    propretion_df = test_df.loc[:,max_sub_nodes]
    max_sub_net = test_df.loc[max_sub_nodes,max_sub_nodes]
    propretion_df.apply(lambda x:get_cell_entroph(x,max_sub_net)/caculate_MaxSr(max_sub_net),axis=1)

else:
    print('ok!,1,2,3,let`s jam!')
