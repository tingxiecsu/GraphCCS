# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:27:49 2023

@author: ZNDX002
"""
import yaml
import json
import numpy as np
import pandas as pd
import bisect
import pubchempy as pc
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
from train import Predict
from GNN_RT_predict import predict_rt

def search_formula(adduct,formulaDB,mass, ppm): 
    if adduct == '[M+NH4]+':
        mass -= ExactMolWt(Chem.MolFromSmiles('[NH4+]'))
    elif adduct == '[M+H]+':
        mass -= ExactMolWt(Chem.MolFromSmiles('[H+]'))
    elif adduct == '[M+Hac-H]-' or adduct == '[M+CH3COO]-':
        mass -= ExactMolWt(Chem.MolFromSmiles('CC(=O)[O-]'))
    elif adduct == '[M-H]-':
        mass += ExactMolWt(Chem.MolFromSmiles('[H+]'))
    mmin = mass - mass*ppm/10**6 
    mmax = mass + mass*ppm/10**6 
    lf = bisect.bisect_left(formulaDB['Exact mass'], mmin) 
    rg = bisect.bisect_right(formulaDB['Exact mass'], mmax) 
    formulas = list(formulaDB['Formula'][lf:rg]) 
    return mass,formulas 

def search_structure_from_mass(adduct,structureDB,mass, ppm): 
    if adduct == '[M+NH4]+':
        mass -= ExactMolWt(Chem.MolFromSmiles('[NH4+]'))
    elif adduct == '[M+H]+':
        mass -= ExactMolWt(Chem.MolFromSmiles('[H+]'))
    elif adduct == '[M+Hac-H]-' or adduct == '[M+CH3COO]-':
        mass -= ExactMolWt(Chem.MolFromSmiles('CC(=O)[O-]'))
    elif adduct == '[M-H]-':
        mass += ExactMolWt(Chem.MolFromSmiles('[H+]'))
    structures=pd.DataFrame()
    mmin = mass - mass*ppm/10**6 
    mmax = mass + mass*ppm/10**6 
    structures = structureDB[(structureDB['MonoisotopicMass'] >= mmin) & (structureDB['MonoisotopicMass'] <= mmax)]
    return mass,list(structures['SMILES'])

def search_pubchem(formula, timeout=9999):
    # get pubchem cid based on formula
    cids = pc.get_cids(formula, 'formula', list_return='flat')
    idstring = ''
    smiles = []
    inchikey = []
    all_cids = []
    # search pubchem via formula with pug
    for i, cid in enumerate(cids):
        idstring += ',' + str(cid)
        if ((i%100==99) or (i==len(cids)-1)):
            url_i = "http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + idstring[1:(len(idstring))] + "/property/InChIKey,CanonicalSMILES/JSON"
            res_i = requests.get(url_i, timeout=timeout)
            soup_i = BeautifulSoup(res_i.content, "html.parser")
            str_i = str(soup_i)
            properties_i = json.loads(str_i)['PropertyTable']['Properties']
            idstring = ''
            for properties_ij in properties_i:
                smiles_ij = properties_ij['CanonicalSMILES']
                if smiles_ij not in smiles:
                    smiles.append(smiles_ij)
                    inchikey.append(properties_ij['InChIKey'])
                    all_cids.append(str(properties_ij['CID']))
                else:
                    wh = np.where(np.array(smiles)==smiles_ij)[0][0]
                    all_cids[wh] = all_cids[wh] + ', ' + str(properties_ij['CID'])
    result = pd.DataFrame({'InChIKey': inchikey, 'SMILES': smiles, 'PubChem': all_cids})
    return result

def score_mz(mass,candidate_list,TOL_min=0.00002,TOL_max=0.00003):
    mols = [Chem.MolFromSmiles(smi) for smi in candidate_list]
    pred_ms = np.array([ExactMolWt(mol) for mol in mols])
    ms_score= []
    relative_error = (abs(pred_ms-mass))/mass
    for error in relative_error:
        if error < TOL_min:
            score = 1
        elif error > TOL_max:
            score = 0
        elif (error >= TOL_min) and (error <= TOL_max):
            score = 1-(error - TOL_min)/(TOL_max - TOL_min)
        ms_score.append(score)
    return ms_score

def score_rt(rt,candidate_list,rt_model_path,TOL_min=0.02,TOL_max=0.15):
    pred_rt=predict_rt(candidate_list,rt_model_path)
    rt_score= []
    relative_error = (abs(pred_rt-rt))/rt
    for error in relative_error:
        if error < TOL_min:
            score = 1
        elif error > TOL_max:
            score = 0
        elif (error >= TOL_min) and (error <= TOL_max):
            score = 1-(error - TOL_min)/(TOL_max - TOL_min)
        rt_score.append(score)
    return rt_score

def score_ccs(ccs,adduct,candidate_list,config_path,model_path,TOL_min=0.015,TOL_max=0.05):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    dataset = pd.DataFrame({'SMILES':candidate_list,'Adduct':[adduct]*len(candidate_list)})
    model_predict = Predict(dataset,model_path,**config)
    pred_ccs,dataset=model_predict.ccs_predict()
    pred_ccs =np.array(pred_ccs)
    ccs_score= []
    relative_error = (abs(pred_ccs-ccs))/ccs
    for error in relative_error:
        if error < TOL_min:
            score = 1
        elif error > TOL_max:
            score = 0
        elif (error >= TOL_min) and (error <= TOL_max):
            score = 1-(error - TOL_min)/(TOL_max - TOL_min)
        ccs_score.append(score)
    return ccs_score

def rank(candidate_list,ms_score,rt_score,ccs_score):
    score=0.4*np.array(ms_score)+0.3*np.array(rt_score)+0.3*np.array(ccs_score)
    candidate_file = pd.DataFrame({'SMILES':candidate_list,'ms_score':ms_score,'rt_score':rt_score,'ccs_score':ccs_score,'score':score})
    candidate_file = candidate_file.sort_values(by='score', ascending=False)
    cout = 1
    candidate_file2 = pd.DataFrame(columns = ['rank'])
    lst = list(set(candidate_file['score']))
    n= len(lst)
    for x in range(n-1):
       for y in range(n-1-x):
          if lst[y]<lst[y+1]:
             lst[y],lst[y+1]=lst[y+1],lst[y]
    for score in lst:
        sub = candidate_file.loc[candidate_file['score'] == score]
        sub['rank'] = cout
        candidate_file2 = pd.concat([candidate_file2,sub],join = 'outer')
        cout += len(sub)
    return candidate_file2

if __name__ == "__main__":

    formulaDB=pd.read_csv('/data/formulaDB.csv')
    lipidblast=pd.read_csv('/data/LipidBlast.csv')
    query_smiles = 'CC\C=C/C\C=C/C\C=C/CCCCCCCC(=O)OC1CCC2(C)C3CCC4(C)C(CCC4C3CC=C2C1)C(C)CCCC(C)C'
    ms = 664.60303
    rt = 14.606
    ccs = 280.4485
    adduct = '[M+NH4]+'
    '''ms,formulas=search_formula(adduct,formulaDB,ms,10)
    pubchem_search_res=pd.DataFrame()
    for formula in tqdm(formulas):
        pubchem_result = search_pubchem(formula)
        pubchem_search_res = pd.concat([pubchem_search_res,pubchem_result])
    candidate_list=list(pubchem_search_res['SMILES])'''
    ms,candidate_list = search_structure_from_mass(adduct,lipidblast,ms,5)
    ms_score = score_mz(ms,candidate_list)
    rt_score = score_rt(rt,candidate_list,'/model/')
    ccs_score = score_ccs(ccs,adduct,candidate_list,config_path='/config/config.yaml',model_path='=/graphccs_model.pt')
    rank_results = rank(candidate_list,ms_score,rt_score,ccs_score)
    print(list(rank_results.loc[rank_results['SMILES'] == query_smiles]['rank'])[0])
