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
from torch import nn 
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable
from dataset import data_process_loader_Property_Prediction
from model import Graphccs
from dataset import featurize_atoms,edit_adduct_mol
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
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

class Graphccs(nn.Sequential):
	def __init__(self,in_feats, hidden_feats=None, activation=None, predictor_dim=None):
		super(Graphccs, self).__init__()
		self.input_dim_drug = 256

		self.gnn = GCN(in_feats=in_feats,
                        hidden_feats=hidden_feats,
                        activation=activation
                        )
		gnn_out_feats = self.gnn.hidden_feats[-1]
		self.readout = WeightedSumAndMax(gnn_out_feats)
		self.transform = nn.Linear((self.gnn.hidden_feats[-1] * 2), predictor_dim)

		self.dropout = nn.Dropout(0.1)

		self.hidden_dims = [1024,1024,512]
		layer_size = len(self.hidden_dims)
		dims = [self.input_dim_drug] + self.hidden_dims

		self.predictor = nn.ModuleList([nn.Linear((dims[i]), dims[i+1]) for i in range(layer_size)])
		self.properte = nn.Linear(self.hidden_dims[-1],1)

	def forward(self, v_D):
		# each encoding
		feats = v_D.ndata.pop('h') 
		node_feats = self.gnn(v_D, feats)
		graph_feats = self.readout(v_D, node_feats)
		v_f = self.transform(graph_feats)
		# concatenate and classify
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1): 
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		v_f = self.properte(v_f)
		return v_f

def graph_calculation(df):
    print("calculating molecular graphs")
    node_featurizer = featurize_atoms
    fc = partial(mol_to_bigraph, add_self_loop=True)
    df['Graph'] = ''
    for i in tqdm(range(len(df['SMILES']))):
        smi = df.loc[i,'SMILES']
        add = df.loc[i,'Adduct']
        mol = Chem.MolFromSmiles(smi)
        v_ds = edit_adduct_mol(mol, add)
        v_d = fc(mol = v_ds, node_featurizer = node_featurizer, edge_featurizer = None,explicit_hydrogens = True)
        df.loc[i,'Graph']=v_d
    return df


class Train():
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
		model= Graphccs(in_feats = self.config['node_feat_size'], hidden_feats = [self.config['gnn_hid_dim_drug']] * self.config['gnn_num_layers'], 
					activation = [F.relu] * self.config['gnn_num_layers'], 
					predictor_dim = self.config['hidden_dim_drug'])
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
    def __init__(self,device,df_data,model,**config):
        self.df_data = df_data
        self.model = model
        self.config = config
        self.device = device

    def ccs_predict(self):
        info = data_process_loader_Property_Prediction(self.df_data.index.values, self.df_data.Label.values, self.df_data, **self.config)
        self.model.to(self.device)
        params = {'batch_size': self.config['batch_size'],
                  'shuffle': False,
                  'num_workers': self.config['num_workers'],
                  'drop_last': False,
                  'sampler':SequentialSampler(info),
                  'collate_fn': dgl_collate_func}
        generator = data.DataLoader(info, **params)
        y_pred = []
        self.model.eval()
        for i, (v_d) in enumerate(generator):
            v_d = v_d.to(self.device)
            score = self.model(v_d)
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





