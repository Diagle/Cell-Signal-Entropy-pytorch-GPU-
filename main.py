import os
import scanpy as sc

from entropy import *
from preprocessing import *


work_space = ''
h5ad_path = ''
net_path = ''

os.getcwd(work_space)

# 读取转录谱
adata = sc.read_h5ad(h5ad_path)
data = adata.to_df()
label = adata.obs
genes = list(data.columns)
samples = list(data.index)

#读取互作网络
net = pd.read_csv(net_path,index_col=0)
net[net>1] = 1

# 提取蛋白互作网络与转录谱共享基因
shared_genes =  get_congenes(data,net)
shared_net  = net.loc[shared_genes,shared_genes]
data1 = data.loc[:,shared_genes]

# 提取共享基因网络的最大连接子网 
max_sub_nodes = extract_max_sub_nodes(shared_net)
data2 = data1.loc[:,max_sub_nodes]
max_sub_net = shared_net.loc[max_sub_nodes,max_sub_nodes]

# 计算Max_Sr
Max_Sr = caculate_MaxSr(max_sub_net,'GPU')

# 计算熵
entrophy_list = []
from rich.progress import track
    
for i in track(data2.index, description="Processing..."):
    entrophy_list.append(get_cell_entroph(data2.loc[i],max_sub_net,'GPU')/Max_Sr)

label[str(net_path)+':Sr'] = entrophy_list
label[str(net_path)+':Max_Sr'] = [Max_Sr]*len(data2)

# 保存结果
label.to_csv('./result.csv')