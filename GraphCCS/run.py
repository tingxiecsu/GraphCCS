# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:23:06 2022

@author: ZNDX002
"""
from train import Train
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt

def train_test(dataset,seed):
    columns=dataset.columns
    N=len(dataset['SMILES'])
    if N<=10:
        pass
    else:
        P=round(N*0.9)
    seed =seed
    np.random.seed(seed)
    train_indices = list(np.random.choice(N,P, replace=False) )
    test_indices= list(np.array(list(set(range(N)) - set(train_indices))) )
    dataset_train = dataset.loc[train_indices,columns]
    dataset_test = dataset.loc[test_indices,columns]
    return dataset_train,dataset_test

def train_val(dataset,seed):
    columns=dataset.columns
    N=list(dataset['id'])
    seed =seed
    np.random.seed(seed)
    train_indices = list(np.random.choice(N,round(len(N)*0.9), replace=False) )
    val_indices= list(np.array(list(set(N) - set(train_indices))))
    dataset_train = dataset.loc[train_indices,columns]
    dataset_val = dataset.loc[val_indices,columns]
    return dataset_train,dataset_val

def data_process(X = None, y=None,Adduct=None, random_seed = 1):
    print('Property Prediction Mode...')
    df_data = pd.DataFrame(zip(X, y,Adduct))
    df_data.rename(columns={0:'SMILES',1: 'Label',2: 'Adduct'},inplace=True)
    print('in total: ' + str(len(df_data)) + ' mols')
    print('unique mols: ' + str(len(df_data['SMILES'].unique())))
    df_data['id'] = range(len(df_data['SMILES']))
    train, test = train_test(df_data,seed=100)
    train,val = train_val(train,seed=100)
    print('Done.')
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def test_plot(res,config):
    r2 = r2_score(res['Label'], res['predict'])
    mae = mean_absolute_error(res['Label'], res['predict'])
    medae = median_absolute_error(res['Label'], res['predict'])
    rmae = np.mean(np.abs(res['Label'] - res['predict']) / res['Label']) * 100
    median_re = np.median(np.abs(res['Label'] - res['predict']) / res['Label'])
    mean_re=np.mean(np.abs(res['Label'] - res['predict']) / res['Label'])
    plt.plot(res ['Label'], res['predict'], '.', color = 'blue')
    plt.plot([0,500], [0,500], color ='red')
    plt.ylabel('Predicted CCS')
    plt.xlabel('Experimental CCS')
    plt.text(0,500, 'R2='+str(round(r2,4)), fontsize=10)
    plt.text(180,500,'MAE='+str(round(mae,4)),fontsize=10)
    plt.text(0, 450, 'MedAE='+str(round(medae,4)), fontsize=10)
    plt.text(180, 450, 'MRE='+str(round(mean_re,4)), fontsize=10)
    plt.text(0, 400, 'MedRE='+str(round(median_re,4)), fontsize=10)
    plt.savefig(config['result_folder']+'p-c.png',dpi=300)
    plt.show()

def main():
    config = yaml.load(open("/config/config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = pd.read_csv('/data/ccsbase_4_2.csv')
    smiles = np.array(list(dataset['SMI']))
    y = np.array(list(dataset['CCS']))
    add = np.array(list(dataset['Adduct']))
    train, val, test = data_process(X = smiles, y = y, Adduct = add,
                                   random_seed = 1)
    graphccs = Train(train,val,test,**config)
    graphccs.train_()
    path = config['result_folder']
    train.to_csv(path+'train.csv',index=False)
    val.to_csv(path+'val.csv',index=False)
    test['predict']=''
    a=list(np.load(config['result_folder'] +'DGL_GCN_logits.npy'))
    for i in range(len(test['SMILES'])):
        test['predict'][i]=a[i]
    test.to_csv(path+'test.csv',index=False)
    test_plot(test,config)

if __name__ == "__main__":
    main()