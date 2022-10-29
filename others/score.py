# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:27:34 2021

@author: missbeautyandyouth
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import rdkit
from rdkit.Chem.Descriptors import ExactMolWt
target_mols = pd.read_csv('...data/filtering.csv')
label_CCS = np.array(list(target_mols['Average CCS']))
label_RT = np.array(list(target_mols['Average Rt(min)']))
label_mz = np.array(list(target_mols['Average Mz']))
adduct=list(target_mols['Adduct type'])
TOL_min = 0.05
TOL_max = 0.015
TOL_min_rt = 0.02
TOL_max_rt = 0.15
TOL_min_mz = 0.00002
TOL_max_mz = 0.00005

for i in tqdm(1960):
    candidate = pd.read_csv('.../predict/' + str(i) + '.csv')
    candidate['predict_mz']=''
    smiles = list(candidate['SMILES'])
    add = adduct[i]
    for s in range(len(candidate['SMILES'])):
        smi=smiles[s]
        mol= Chem.MolFromSmiles(smi)
        cal_mass = ExactMolWt(mol)
        if add == '[M+NH4]+':
            cal_mass += ExactMolWt(Chem.MolFromSmiles('[NH4+]'))
        elif add == '[M+H]+':
            cal_mass += ExactMolWt(Chem.MolFromSmiles('[H+]'))
        elif add == '[M+Hac-H]-' or add == '[M+CH3COO]-':
            cal_mass += ExactMolWt(Chem.MolFromSmiles('CC(=O)[O-]'))
        elif add == '[M-H]-':
            cal_mass -= ExactMolWt(Chem.MolFromSmiles('[H+]'))
        candidate.loc[s,'predict_mz'] = cal_mass

    candidate['CCS_score'] = ''
    candidate['RT_score'] = ''
    candidate['mz_score'] = ''
    pred_CCS = np.array(list(candidate['predict']))
    pred_RT = np.array(list(candidate['predict_rt']))
    pred_mz = np.array(list(candidate['predict_mz']))
    relative_error = (abs(pred_CCS-label_CCS[i]))/label_CCS[i]
    relative_error_rt = (abs(pred_RT-label_RT[i]))/label_RT[i]
    relative_error_mz = (abs(pred_mz-label_mz[i]))/label_mz[i]

    for j in len(relative_error):
        if relative_error[j] < TOL_min:
            ccs_score = 1
        elif relative_error[j] > TOL_max:
            ccs_score = 0
        elif (relative_error[j] >= TOL_min) and (relative_error[j] <= TOL_max):
            ccs_score = 1-(relative_error[j] - TOL_min)/(TOL_max - TOL_min)
        candidate.loc[j,'CCS_score'] = ccs_score
    for k in len(relative_error_rt):
        if relative_error_rt[k] < TOL_min_rt:
            rt_score = 1
        elif relative_error_rt[k] > TOL_max_rt:
            rt_score = 0
        elif (relative_error_rt[k] >= TOL_min_rt) and (relative_error_rt[k] <= TOL_max_rt):
            rt_score = 1-(relative_error_rt[k] - TOL_min_rt)/(TOL_max_rt - TOL_min_rt)
        candidate.loc[k,'RT_score'] = rt_score
    for l in len(relative_error_mz):
        if relative_error_mz[l] < TOL_min_mz:
            mz_score = 1
        elif relative_error_mz[l] > TOL_max_mz:
            mz_score = 0
        elif (relative_error_mz[l] >= TOL_min_mz) and (relative_error_mz[l] <= TOL_max_mz):
            mz_score = 1-(relative_error_mz[l] - TOL_min_mz)/(TOL_max_mz - TOL_min_mz)
        candidate.loc[l,'mz_score'] = mz_score
