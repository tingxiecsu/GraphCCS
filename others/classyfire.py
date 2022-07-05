# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:43:58 2021

@author: -
"""
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from rdkit import Chem
from tqdm import tqdm
data = pd.read_csv('/ccsbase_4_1.csv')
SMIS = list(data['SMI'])
data['class'] = ''
data['subclass'] = ''
for i,smi in enumerate(tqdm(SMIS)):
    mol = Chem.MolFromSmiles(smi)
    inchi_key = Chem.MolToInchiKey(mol)
    url = 'http://classyfire.wishartlab.com/entities/' + inchi_key
    time.sleep(3)
    strhtml = requests.get(url,verify=False)
    soup = BeautifulSoup(strhtml.text,'lxml')
    d = soup.select('div:nth-child(9) > p:nth-child(2)')
    d2 = soup.select('div:nth-child(11) > p:nth-child(2)')
    for item in d:
        result = {
            'class':item.get_text(),
            'link':item.get('href')
            }
    clas = result['class']
    data['class'][i] = clas   
    for item in d2:
        result2 = {
            'subclass':item.get_text(),
            'link':item.get('href')
            }
    subclass = result2['subclass']
    data['subclass'][i] = subclass
