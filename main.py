import os
from matplotlib.backend_tools import SaveFigureBase
import scanpy as sc
import sys
sys.path.append('/home/jinyuanxu/WorkSpace/hugang/entropy/Cell-Signal-Entropy-pytorch-GPU-')
from entropy import *
from preprocessing import *
from typing import Optional,Union,Any,Set,List,Tuple,Dict

def run_main(df=pd.DataFrame,net=pd.DataFrame,device:str='GPU'):
    '''
    输入存储目录,单细胞h5ad,ppi网络邻接矩阵,方法与设备
    返回结果存储在result.csv中

    '''
    data = df
    net = net
    device = device

    if not torch.cuda.is_available():
        print('GPU cuda is not available, it is running on the CPU!')

    # 矩阵处理
    genes = list(data.columns)
    samples = list(data.index)

    print(f'RNA data has {len(samples)} cells and {len(genes)} genes')


    # 互作网络处理
    net[net>1] = 1
    print(f'net has {len(net)} nodes and {np.sum(net.values)} edges')


    # 提取蛋白互作网络与转录谱共享基因
    shared_genes =  get_congenes(data,net)
    shared_net  = net.loc[shared_genes,shared_genes]
    data1 = data.loc[:,shared_genes]
    print(f'ppi and matrix shared {len(shared_genes)} genes')

    # 提取共享基因网络的最大连接子网 
    max_sub_nodes = extract_max_sub_nodes(shared_net)
    data2 = data1.loc[:,max_sub_nodes]
    max_sub_net = shared_net.loc[max_sub_nodes,max_sub_nodes]
    print(f'max sub net has {len(max_sub_net)} nodes and {np.sum(max_sub_net.values)} edges')


    # 计算Max_Sr
    Max_Sr = caculate_MaxSr(max_sub_net,device)
    print(f'Max_sr:{Max_Sr:.2f}')

    # 计算熵
    entrophy_list = []
    from rich.progress import track
    for i in track(data2.index, description="Processing..."):
        entrophy_list.append(get_cell_entroph(data2.loc[i],max_sub_net,device)/Max_Sr)

    return entrophy_list


if __name__ == '__main__':

    work_space = '/home/jinyuanxu/WorkSpace/hugang/entropy/GSE114687/MCF10A/'
    os.chdir(work_space)

    h5ad_path = './filter_data.h5ad'
    adata = sc.read_h5ad(h5ad_path)
    print('adata load')
    data = adata.to_df()
    label = adata.obs


    # 读取网络
    nets = {
        'net_string700': pd.read_csv('../../STRING_0.7_homo.csv',index_col=0),
        'net_string' : pd.read_csv('../../string.csv',index_col=0),
        'net17' : pd.read_csv('../../net17.csv' ,index_col=0),
        'net19' : pd.read_csv('../../net19.csv',index_col=0),}
    device = 'GPU'

    for i in nets.keys():
        label[i] = run_main(data,nets[i],device)
    label.to_csv('./result.csv')

else:
    print('main load successly!')

