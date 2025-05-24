import pandas as pd
import numpy as np
from typing import Optional,Union,Any,Set,List,Tuple,Dict

def get_congenes(df:pd.DataFrame,net:pd.DataFrame)->List:
    con_genes = list(set(df.columns).intersection(net.columns))
    return con_genes

def gene_id_transform(gene_list:List,old_form:str='entrezgene', new_form:str='symbol', species='human')->List:
    '''
    转换的基因格式
    entrezgene,uniprot,symbol,accession
    '''
    from mygene import MyGeneInfo
    mg = MyGeneInfo()
    gene_list = [str(i) for i in gene_list]
    res = mg.querymany(gene_list,scopes=old_form,fileds=new_form,returnall=False)
    df = pd.DataFrame(res)
    trans = {df['query'][i]:df[new_form][i] for i in df.index}
    return [str(trans[i]).upper() for i in gene_list]

def drop_multi(df:pd.DataFrame,axis:int,method='mean')->pd.DataFrame:
    '''
    去除重复行或列，方法为'mean','first'
    '''
    if method =='mean':
        if axis == 0:
            return df.groupby(by = df.index,axis=0).mean()
        else:
            return df.groupby(by = df.columns,axis=1).mean()    
    elif method =='first':
        if axis == 0:
            return df.groupby(by = df.index,axis=0).head(1)
        else:
            return df.groupby(by = df.columns,axis=1).head(1)  

def normalized_z_score(x):
    return (x - x.mean(axis=0))/x.std(axis=0)
def normalized_log1plus(x):
    return np.log1p(x)
def normalized_minmax(x):
    return (np.max(x,axis=0)-x)/np.max(x,axis=0)
def normalized_sum(x):
    return (x.T/np.sum(x,axis=1)).T*1e4
def normalized_index(x):
    return (np.e ** x -1)

def get_hvgs(x,rate=0.05):
    if rate == 0:
        return list(x.columns)
    elif rate == 2000:
        return list(np.var(x,axis=0).sort_values(ascending=False).index[:2000])
    hvgs = list(np.var(x,axis=0).sort_values(ascending=False).index[:int(x.shape[1]*rate)])
    return hvgs


def normal_hvgs(x:pd.DataFrame,method:Optional[str]=None,rate:Optional[Union[int,float]]=0)->pd.DataFrame:
    '''
    log1p,zscore,minmax,sum,index
    '''
    hvgs = get_hvgs(x,rate)
    if method == 'log1p':
        x1 = normalized_log1plus(x)
    elif method == 'z_score' :
        x1 = normalized_z_score(x)
    elif method == 'minmax':
        x1 = normalized_minmax(x)
    elif method == 'sum':
        x1 = normalized_sum(x)
    elif method == 'index':
        x1 = normalized_index(x)
    else:
        x1 = x
    return x1[hvgs]


if __name__ =='__main__':
    print('再报错让你码飞起来')
    gene_list=['ENSMUSG00000000001','ENSMUSG00000000003','ENSMUSG00000000028','ENSMUSG00000000031','ENSMUSG00000000037']
    gene_id_transform(gene_list)

else:
    print('加载成功')