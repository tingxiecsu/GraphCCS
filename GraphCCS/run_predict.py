# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:44:15 2022

@author: ZNDX002
"""

from train import Predict
import yaml
import pandas as pd


def main():
    config = yaml.load(open("/config/config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = pd.read_csv('/data/')
    model_path = '/model/model.pt'
    model_predict = Predict(dataset,model_path,**config)
    y=model_predict.ccs_predict()
    dataset['predicts']=''
    for i in range(len(dataset['SMILES'])):
        dataset['predicts'][i]=y[i]

if __name__ == "__main__":
    main()
