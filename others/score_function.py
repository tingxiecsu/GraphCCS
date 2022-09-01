# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:27:34 2021

@author: missbeautyandyouth
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
real_mols = pd.read_csv('/Mouse_Adrenal_glad.csv')
label_CCS = np.array(list(real_mols['Average CCS']))
label_RT = np.array(list(real_mols['Average Rt(min)']))
TOL_min = 0.0005
TOL_max = 0.03
TOL_min_rt = 0.005
TOL_max_rt = 0.05
for i in tqdm(760):
    candidate = pd.read_csv('/predict/' + str(i) + '.csv')
    candidate['CCS_score'] = ''
    candidate['RT_score'] = ''
    pred_CCS = np.array(list(candidate['predict']))
    pred_RT = np.array(list(candidate['predict_rt']))
    relative_error = (abs(pred_CCS-label_CCS[i]))/label_CCS[i]
    relative_error_rt = (abs(pred_RT-label_RT[i]))/label_RT[i]
    for j in len(relative_error):
        if relative_error[j] < TOL_min:
            ccs_score = 1
        elif relative_error[j] > TOL_max:
            ccs_score = 0
        elif (relative_error[j] >= TOL_min) and (relative_error[j] <= TOL_max):
            ccs_score = 1-(relative_error[j] - TOL_min)/(TOL_max - TOL_min)
        candidate.loc[j,'CCS_score'] = ccs_score
    for k in len(relative_error_rt):
        if relative_error_rt[j] < TOL_min_rt:
            rt_score = 1
        elif relative_error_rt[j] > TOL_max_rt:
            rt_score = 0
        elif (relative_error_rt[j] >= TOL_min_rt) and (relative_error_rt[j] <= TOL_max_rt):
            rt_score = 1-(relative_error_rt[j] - TOL_min_rt)/(TOL_max_rt - TOL_min_rt)
        candidate.loc[k,'RT_score'] = rt_score
candidate.to_csv('/score/' + str(i) + '.csv')