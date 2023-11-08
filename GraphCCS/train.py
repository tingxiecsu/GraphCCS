# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:15:35 2022

@author: ZNDX002
"""

import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
import os
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable
from dataset import data_process_loader_Property_Prediction,data_process_loader_Property_Prediction
from model import Graphccs
from dataset import featurize_atoms,edit_adduct_mol
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, BaseBondFeaturizer
from dgllife.utils import ConcatFeaturizer,bond_type_one_hot, bond_is_in_ring_one_hot,bond_is_conjugated_one_hot,bond_stereo_one_hot,bond_direction_one_hot
from functools import partial
from tqdm import tqdm
from rdkit import Chem

def save_dict(path, obj):
	with open(os.path.join(path, 'config.pkl'), 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

 
def dgl_collate_func(x):
	x, y= zip(*x)
	import dgl
	x = dgl.batch(x)
	return x, torch.tensor(y)

def dgl_predict_collate_func(x):
	import dgl
	x = dgl.batch(x)
	return x

class BondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_field='e', self_loop=False):
        super(BondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated_one_hot,
                 bond_is_in_ring_one_hot,
                 bond_stereo_one_hot,
                 bond_direction_one_hot]
            )}, self_loop=self_loop)

def graph_calculation(df):
    print("calculating molecular graphs")
    node_featurizer = featurize_atoms
    edge_featurizer = BondFeaturizer(bond_data_field='e', self_loop=True)
    fc = partial(mol_to_bigraph, add_self_loop=True)
    df['Graph'] = ''
    for i in tqdm(range(len(df['SMILES']))):
        smi = df.loc[i,'SMILES']
        add = df.loc[i,'Adduct']
        mol = Chem.MolFromSmiles(smi)
        v_ds = edit_adduct_mol(mol, add)
        v_d = fc(mol = v_ds, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer,explicit_hydrogens = True)
        df.loc[i,'Graph']=v_d
    return df


class Train():
	"""
    Train, validate, and test the model
    Arguments:
        train: pandas.core.frame.DataFrame
            training set.
        val: pandas.core.frame.DataFrame
            validation set.
        test: pandas.core.frame.DataFrame
            testing set.
        config: dict
            config dictionary.
	"""
	def __init__(self,  train, val, test,**config):
		self.config = config
		self.train = train
		self.test = test
		self.val = val
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.result_folder = config['result_folder']
		if not os.path.exists(self.result_folder):
			os.mkdir(self.result_folder)
		self.binary = False

	def test_(self, data_generator, model):
		"""
	test the model
	Arguments:
		data_generator: torch.utils.data.dataloader.DataLoader
				testing generator
        model: model
				GraphCCS model.
	Returns:
		means quared error of the predicted values and labels
		Pearson's correlation coefficient of the predicted values and labels
		Pearson's correlation coefficient with p-value of the predicted values and labels
		concordance index of the predicted values and labels
		y_pred: list
				predicted CCS values
	"""
		y_pred = []
		y_label = []
		model.eval()
		for i, (v_d, label) in enumerate(data_generator):
			v_d = v_d.to(self.device)
			score = model(v_d)
			logits = torch.squeeze(score).detach().cpu().numpy()
			label_ids = label.to('cpu').numpy()
			y_label = y_label + label_ids.flatten().tolist()
			y_pred = y_pred + logits.flatten().tolist()
		model.train()
		return mean_squared_error(y_label, y_pred), \
			pearsonr(y_label, y_pred)[0], \
			pearsonr(y_label, y_pred)[1], \
			concordance_index(y_label, y_pred), y_pred

	def train_(self):
		model=GraphCCS(node_in_dim=self.config['node_feat_size'], edge_in_dim=self.config['edge_feat_size'], 
											hidden_feats=[self.config['hid_dim']]*self.config['num_layers'], 
											gru_out_layer=self.config['gru_out_layer'],  
											dropout=self.config['dropout'], residual=True)
		model = model.to(self.device)
		lr = self.config['LR']
		decay = self.config['decay']
		BATCH_SIZE = self.config['batch_size']
		train_epoch = self.config['train_epoch']
		decay_interval = self.config['decay_interval']
		loss_history = []
		opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-6)
		info_test = data_process_loader_Property_Prediction(self.test.index.values, self.test.Label.values, self.test)
		params = {'batch_size': BATCH_SIZE,
				'shuffle': True,
				'num_workers': self.config['num_workers'],
				'drop_last': False,
			'collate_fn': dgl_collate_func}
		params_test = {'batch_size': BATCH_SIZE,
					'shuffle': False,
					'num_workers': self.config['num_workers'],
					'drop_last': False,
					'sampler':SequentialSampler(info_test),
					'collate_fn': dgl_collate_func}
		train = graph_calculation(self.train)
		test = graph_calculation(self.test)
		val = graph_calculation(self.val)

		training_generator = data.DataLoader(data_process_loader_Property_Prediction(train.index.values,  train.Label.values,train), **params)
		validation_generator = data.DataLoader(data_process_loader_Property_Prediction(val.index.values, val.Label.values, val), **params)
		testing_generator = data.DataLoader(data_process_loader_Property_Prediction(test.index.values, test.Label.values, test), **params_test)

		max_MSE = 10000
		model_max = copy.deepcopy(model)
		valid_metric_record = []
		valid_metric_header = ["# epoch"] 
		valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
		table = PrettyTable(valid_metric_header)
		float2str = lambda x:'%0.4f'%x

		t_start = time() 
		loss_train=[]
		loss_val=[]
		num=[]
		for epo in range(train_epoch):
			train_loss = 0
			counter = 0
			if epo % decay_interval ==0:
				opt.param_groups[0]['lr'] *= 0.85
			for i, (v_d, label) in enumerate(training_generator):
				v_d = v_d.to(self.device)
				score = model(v_d)
				label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)
				loss_fct = torch.nn.MSELoss()
				n = torch.squeeze(score, 1)
				loss = loss_fct(n, label)
				loss_history.append(loss.item())
				opt.zero_grad()
				loss.backward()
				opt.step()
				train_loss += loss.item()
				counter += 1

				if (i % 100 == 0):
					t_now = time()
					print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
						' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
						". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
			num.append(epo + 1)
			train_loss /= counter
			loss_train.append(train_loss)

			with torch.set_grad_enabled(False):
				mse, r2, p_val, CI, logits = self.test_(validation_generator, model)
				lst = ["epoch " + str(epo)] + list(map(float2str,[mse, r2, p_val, CI]))
				valid_metric_record.append(lst)
				loss_val.append(mse)
				if mse < max_MSE:
					model_max = copy.deepcopy(model)
					max_MSE = mse
					print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
					       + str(r2)[:7] + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI)[:7])
			table.add_row(lst)
		np.save(self.config['result_folder']+'loss_train.npy',loss_train)
		np.save(self.config['result_folder']+'loss_val.npy',loss_val)
		prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
		with open(prettytable_file, 'w') as fp:
			fp.write(table.get_string())

		model = model_max
		self.save_model(model,self.config['result_folder'])
		print('--- Go for Testing ---')
		mse, r2, p_val, CI, logits = self.test_(testing_generator, model_max)
		test_table = PrettyTable(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
		test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
		print('Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
			+ ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
		np.save(os.path.join(self.result_folder, str('DGL_GCN')+ '_logits.npy'), np.array(logits))                
		prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
		with open(prettytable_file, 'w') as fp:
			fp.write(test_table.get_string())

		fontsize = 16
		iter_num = list(range(1,len(loss_history)+1))
		plt.figure(3)
		plt.plot(iter_num, loss_history, "bo-")
		plt.xlabel("iteration", fontsize = fontsize)
		plt.ylabel("loss value", fontsize = fontsize)
		pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
		with open(pkl_file, 'wb') as pck:
			pickle.dump(loss_history, pck)

		fig_file = os.path.join(self.result_folder, "loss_curve.png")
		plt.savefig(fig_file)

	def save_model(self,model, path_dir):
		if not os.path.exists(path_dir):
			os.makedirs(path_dir)
		torch.save(model.state_dict(), path_dir + '/model.pt')
		save_dict(path_dir, self.config)


class Predict():
    """
    Predict CCS values
    Arguments:
        df_data: pandas.core.frame.DataFrame
            Datasets with SMILES and adduct types of compounds to be predicted.
        model_path: string
            path to the pretrained model.
        config: dict
            config dictionary.
    Returns:
        y_pred: list
            predicted CCS values
    """
    def __init__(self,df_data,model_path,**config):
        self.df_data = df_data
        self.model_path = model_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def ccs_predict(self):
        model=GraphCCS(node_in_dim=self.config['node_feat_size'], edge_in_dim=self.config['edge_feat_size'], 
                                         hidden_feats=[self.config['hid_dim']]*self.config['num_layers'],
                                         gru_out_layer=self.config['gru_out_layer'], 
                                         dropout=self.config['dropout'], residual=True)
        load_pretrained( model, self.model_path, device = self.device)
        info = data_process_loader_Property_Prediction(self.df_data.index.values, self.df_data)
        model.to(self.device)
        params = {'batch_size': self.config['batch_size'],
                  'shuffle': False,
                  'num_workers': self.config['num_workers'],
                  'drop_last': False,
                  'sampler':SequentialSampler(info),
                  'collate_fn': dgl_predict_collate_func}
        generator = data.DataLoader(info, **params)
        y_pred = []
        model.eval()
        for i, (v_d) in enumerate(generator):
            v_d = v_d.to(self.device)
            score = model(v_d)
            logits = torch.squeeze(score).detach().cpu().numpy()
            y_pred = y_pred + logits.flatten().tolist()
        return y_pred


def load_pretrained(device,model, path):
	if not os.path.exists(path):
		os.makedirs(path)
	if device == 'cuda':
		state_dict = torch.load(path)
	else:
		state_dict = torch.load(path, map_location = torch.device('cpu'))

	model.load_state_dict(state_dict)





