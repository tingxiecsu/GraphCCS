# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:20:00 2022

@author: ZNDX002
"""

import json
import numpy as np
import torch
import random
from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem,rdMolDescriptors
from torch.utils import data
from functools import partial
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

class BaseEncoder(object):
    """
    Base encoder class to encode data in a particular way
    """

    def __init__(self):
        self.converter = {}
        self._is_fit = False

    def load_encoder(self, json_file):
        with open(json_file, "r") as fi:
            self.converter = json.load(fi)
        if len(self.converter) > 0:
            self._is_fit = True

    def save_encoder(self, file_name):
        with open(file_name, "w") as fo:
            json.dump(self.converter, fo)

    def fit(self, X):
        self._fit(X)
        self._is_fit = True

    def _fit(self, X):
        pass

    def transform(self, X):
        if self._is_fit:
            return self._transform(X)
        else:
            raise RuntimeError("Encoder must be fit first")

    def _transform(self, X):
        pass

class AdductToOneHotEncoder(BaseEncoder):

    def _fit(self, X):
        """
        X : array of all elements
        """
        for i, j in enumerate(set(X)):
            self.converter[j] = i
    def _transform(self, X):
        X_encoded = np.zeros(len(self.converter))
        X_encoded[self.converter[X]] = 1
        return X_encoded

def atom_to_feature(atom,mol):
    '''calculates atom feature vector
    Parameters
    ----------
    atom : The rdkit.Chem.rdchem.Atom of rdkit.Chem.rdchem.Mol object
    mol  : The rdkit.Chem.rdchem.Mol object
    Returns
    -------
    Feature array of each atom
    '''
    atom_features1 = []
    atom_features2 = []
    atom_list = ['C','N', 'O', 'S', 'F', 'Si', 'P','Cl', 'Br', 'Mg', 'Na',
                 'Ca', 'Fe', 'As', 'Al','I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 
                 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn','H', 'Li', 'Ge', 
                 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr','Pt', 'Hg', 'Pb',
                 'Te','Mo','Nb','He','Be','Ne','Ar','Sc','Ti','Cr','Ga','Ge',
                 'Kr','Rb','Sr','Zr','Mo','Tc','Ru','Rh','Xe','Cs','Ba',
                 'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
                 'Yb','Lu','Hf','Ta','Re','Os','Ir','Bi','Po','At','Rn','Fr',
                 'Ra','Ac','Th','Pa','Np','Pu','Am','Cm','Bk','Cf','Ts','Md','W']
    atom_index = atom.GetIdx()
    AllChem.ComputeGasteigerCharges(mol)
    GasteigerCharge = atom.GetProp('_GasteigerCharge')
    if GasteigerCharge in ['-nan', 'nan', '-inf', 'inf']:
       GasteigerCharge = 0
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    CripperLogP = contribs[atom_index][0]
    MolarRefrac = contribs[atom_index][1]
    atom_symbol_encoder = AdductToOneHotEncoder()
    atom_symbol_encoder.fit(atom_list)
    atom_symbol = atom_symbol_encoder.transform(atom.GetSymbol())
    atom_is_in_ring_encoder = AdductToOneHotEncoder()
    atom_is_in_ring_encoder.fit([True,False])
    atom_ring = atom_is_in_ring_encoder.transform(atom.IsInRing())
    atom_aromatic = atom_is_in_ring_encoder.transform(atom.GetIsAromatic())
    mol = atom.GetOwningMol()
    atom_hacceptor =  atom_is_in_ring_encoder.transform(atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)])
    atom_hdonor = atom_is_in_ring_encoder.transform(atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)])
    atom_hybrid_encoder = AdductToOneHotEncoder()
    atom_hybrid_encoder.fit([Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         Chem.rdchem.HybridizationType.UNSPECIFIED,
                         Chem.rdchem.HybridizationType.S])
    atom_hybrid = atom_hybrid_encoder.transform(atom.GetHybridization())
    atom_numhs_encoder = AdductToOneHotEncoder()
    atom_numhs_encoder.fit([0,1,2,3,4])
    atom_numhs = atom_numhs_encoder.transform(min(atom.GetTotalNumHs(),4))
    atom_valence_encoder=AdductToOneHotEncoder()
    atom_valence_encoder.fit([0,1,2,3,4,5,6])
    atom_valence=atom_valence_encoder.transform(min(atom.GetTotalValence(),6))
    atom_degree_encoder=AdductToOneHotEncoder()
    atom_degree_encoder.fit([0,1,2,3,4,5])
    atom_degree=atom_degree_encoder.transform(min(atom.GetDegree(),5))
    atom_ringsize_encoder=AdductToOneHotEncoder()
    atom_ringsize_encoder.fit([0,3,4,5,6,7,8,9,10])
    for ring_size in [10,9,8,7,6,5,4,3,0]:
        if atom.IsInRingSize(ring_size):
            break
    atom_ringsize = atom_ringsize_encoder.transform(ring_size)
    atom_Chiral = float(atom.HasProp("_ChiralityPossible"))
    atom_mass = float(atom.GetMass()/100)
    asa = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom_index] 
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)[atom_index]
    atom_features1 = np.concatenate((atom_symbol,atom_degree,atom_ring,atom_hacceptor,
                                     atom_hdonor,atom_numhs,atom_aromatic,atom_hybrid,
                                     atom_valence,atom_ringsize),-1)
    atom_features2.append([atom_Chiral,atom_mass,GasteigerCharge,CripperLogP,MolarRefrac,asa,tpsa])
    from sklearn.preprocessing import normalize
    atom_features2 = normalize(atom_features2, axis=1, norm='max')
    atom_features2 = list(atom_features2.squeeze(-2))
    return np.concatenate((atom_features1,atom_features2),-1)

def featurize_atoms(mol):
    """A featurizer for atoms.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem Mol object
    Examples
    --------
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = featurize_atoms
    >>> atom_featurizer(mol)
    {'h': tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               1.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0173, -0.0060,  0.0208,
               0.3615,  1.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
               1.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,
               1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0182,  0.0061, -0.0308,
               0.4167,  1.0000,  0.0000],
             [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               1.0000,  0.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000,
               0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
               0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0079, -0.0196, -0.0143,
               0.0407,  0.2524,  1.0000]])}
    """
    feature = []
    for atom in mol.GetAtoms():
        feature.append(atom_to_feature(atom,mol))
    feature = np.array(feature).reshape(-1,147)
    feature = torch.tensor(feature)
    #feature[:,-7] = feature[torch.randperm(feature.size(0)),-7]
    return {'h': feature.float()}

def edit_adduct_mol(mol,adduct):
    """
    Parameters
    ----------
    mol : The rdkit.Chem.rdchem Mol object
    adduct : The adduct type
    Returns
    -------
    edit_mol : The editted rdkit.Chem.rdchem Mol object
    """
    atoms_num = mol.GetNumAtoms()
    mol_nH = mol
    mol = Chem.AddHs(mol)
    edit_mol = mol
    AllChem.ComputeGasteigerCharges(mol)
    partial_charge=[]
    for atom in mol.GetAtoms():
        GasteigerCharge = atom.GetProp('_GasteigerCharge')
        if GasteigerCharge in ['-nan', 'nan', '-inf', 'inf']:
           GasteigerCharge = 0
        partial_charge.append(float(GasteigerCharge))
    pkas = [partial_charge.index(sorted(partial_charge[:atoms_num])[-i]) for i in range(1,atoms_num)]
    pka = partial_charge.index(max(partial_charge[:atoms_num]))
    pkb = partial_charge.index(min(partial_charge[:atoms_num]))
    if adduct[-1] == '+':
        if adduct == '[M+H]+':
            mod = Chem.MolFromSmiles('[H+]')
            combo = Chem.CombineMols(mol,mod)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(pkb,len(partial_charge),order=Chem.rdchem.BondType.IONIC)
            edit_mol = edcombo.GetMol()
        if adduct == '[M+Na]+':
            mod = Chem.MolFromSmiles('[Na+]')
            combo = Chem.CombineMols(mol,mod)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(pkb,len(partial_charge),order=Chem.rdchem.BondType.IONIC)
            edit_mol = edcombo.GetMol()
        if adduct == '[M+NH4]+':
            mod = Chem.MolFromSmiles('[NH4+]')
            combo = Chem.CombineMols(mol,mod)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(pkb,len(partial_charge),order=Chem.rdchem.BondType.IONIC)
            edit_mol = edcombo.GetMol()
        if adduct == '[M+K]+':
            mod = Chem.MolFromSmiles('[K+]')
            combo = Chem.CombineMols(mol,mod)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(pkb,len(partial_charge),order=Chem.rdchem.BondType.IONIC)
            edit_mol = edcombo.GetMol()
        if adduct == '[M+H-H2O]+':
            GasteigerCharge_initial = 0.0
            leaving_group = mol.GetAtomWithIdx(0)
            H_idx = 0
            for atom in mol_nH.GetAtoms():
                if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() != 0:
                    atom_idx = atom.GetIdx()
                    atom = mol.GetAtomWithIdx(atom_idx)
                    GasteigerCharge = float(atom.GetProp('_GasteigerCharge'))
                    if GasteigerCharge < GasteigerCharge_initial:
                        leaving_group = atom
                        neighbors_idx = [x.GetIdx() for x in atom.GetNeighbors()]
                        H_idx = max(neighbors_idx)
                        GasteigerCharge_intial = GasteigerCharge
            O_idx = leaving_group.GetIdx()
            mw = Chem.RWMol(mol)
            mw.RemoveAtom(H_idx)
            mw.RemoveAtom(O_idx)
            edit_mol = mw.GetMol()
    elif adduct[-1] == '-':
        if adduct == '[M-H]-':
            H_idx = 0
            for pka in pkas:
                atom = mol_nH.GetAtomWithIdx(pka)
                if atom.GetTotalNumHs() != 0:
                    atom = mol.GetAtomWithIdx(pka)
                    neighbors = [x for x in atom.GetNeighbors()]
                    neighbors_idx = [x.GetAtomicNum() for x in atom.GetNeighbors()]
                    if 1 in neighbors_idx:
                        H_idx = neighbors[neighbors_idx.index(1)].GetIdx()
                    mw = Chem.RWMol(mol)
                    mw.RemoveAtom(H_idx)
                    edit_mol = mw.GetMol()
                    break
                else:
                    continue
        if adduct == '[M+HCOO]-':
            mod = Chem.MolFromSmiles('C(=O)[O-]')
            combo = Chem.CombineMols(mol,mod)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(pka,len(partial_charge)+2,order=Chem.rdchem.BondType.IONIC)
            edit_mol = edcombo.GetMol()
        if adduct == '[M+CH3COO]-':
            mod = Chem.MolFromSmiles('CC(=O)[O-]')
            combo = Chem.CombineMols(mol,mod)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(pka,len(partial_charge)+3,order=Chem.rdchem.BondType.IONIC)
            edit_mol = edcombo.GetMol()
        if adduct == '[M+Na-2H]-':
            mod = Chem.MolFromSmiles('[Na+]')
            combo = Chem.CombineMols(mol,mod)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(pkb,len(partial_charge),order=Chem.rdchem.BondType.IONIC)
            edit_mol = edcombo.GetMol()
            for pka in pkas:
                pka = pka
                atom = mol_nH.GetAtomWithIdx(pka)
                if atom.GetTotalNumHs() != 0:
                    atom = edit_mol.GetAtomWithIdx(pka)
                    neighbors = [x for x in atom.GetNeighbors()]
                    neighbors_idx = [x.GetAtomicNum() for x in atom.GetNeighbors()]
                    if 1 in neighbors_idx:
                        H_idx = neighbors[neighbors_idx.index(1)].GetIdx()
                    mw = Chem.RWMol(edit_mol)
                    mw.RemoveAtom(H_idx)
                    edit_mol = mw.GetMol()
                    break
                else:
                    continue
            for pka in pkas[pkas.index(pka)+1:]:
                pka = pka
                atom = mol_nH.GetAtomWithIdx(pka)
                if atom.GetTotalNumHs() != 0:
                    atom = mol.GetAtomWithIdx(pka)
                    neighbors = [x for x in atom.GetNeighbors()]
                    neighbors_idx = [x.GetAtomicNum() for x in atom.GetNeighbors()]
                    if 1 in neighbors_idx:
                        H_idx = neighbors[neighbors_idx.index(1)].GetIdx()
                    mw = Chem.RWMol(edit_mol)
                    mw.RemoveAtom(H_idx)
                    edit_mol = mw.GetMol()
                    break
                else:
                    continue
        if adduct == '[M-CH3]-':
            end_group = []
            for atom in mol.GetAtoms():
                neighbors_idx = [x.GetAtomicNum() for x in atom.GetNeighbors()]
                H_num = neighbors_idx.count(1)
                if atom.GetSymbol() == 'C' and H_num == 3:
                    end_group.append(atom)
            atom_del = random.choice(end_group)
            atom_del_idx = atom_del.GetIdx()
            mw = Chem.RWMol(mol)
            mw.RemoveAtom(atom_del_idx)
            edit_mol = mw.GetMol()
    return edit_mol

class data_process_loader_Property_Prediction(data.Dataset):

    def __init__(self,list_IDs,labels,df):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.df.loc[index,'Graph']
        y = self.labels[index]
        return v_d, y