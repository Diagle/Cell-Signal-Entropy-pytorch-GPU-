import os
import scanpy as sc
from entropy import *
from preprocessing import *
from typing import Optional,Union,Any,Set,List,Tuple,Dict

def main(work_path,h5ad_path:str='None',net_path:str='None',device:str='GPU'):
    '''
    输入工作目录，单细胞h5ad路径，网络路径，方法与设备
    返回结果存储在result.csv中

    '''
    work_space = work_path
    h5ad_path = h5ad_path
    net_path = net_path
    method = method
    device = device

    os.chdir(work_space)

    # 读取转录谱
    adata = sc.read_h5ad(h5ad_path)
    data = adata.to_df()
    label = adata.obs
    genes = list(data.columns)
    samples = list(data.index)

    print(f'RNA data has {len(samples)} cells and {len(genes)} genes')


    #读取互作网络
    net = pd.read_csv(net_path,index_col=0)
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


    label['Sr'] = entrophy_list
    label['Max_Sr'] = [Max_Sr]*len(data2)

    # 保存结果
    label.to_csv('./result.csv')
    print(f'successly run \nresult is saved in {work_path}./result.csv')

    return 0


if __name__ == '__main__':
    work_space = WorkSpace = '/mnt/e/data'
    h5ad_path = './网络熵/singlecell/GSE200981/filter_data.h5ad'
    net_path = './PIN/STRING/string.csv'