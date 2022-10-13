## GraphCCS
This is the code repo for the paper Prediction of Collision Cross Sections with Graph Convolutional
Network for Metabolites Annotation. We developed a method named GraphCCS which can obtain the data-driven
representations of molecules through the end-to-end learning with GCN, and predict the retention time with 
the GCN-learned representations. The network architecture is showed as follow:
![image](https://github.com/tingxiecsu/GraphCCS/blob/main/image/fig4.png)
### Package required:
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/) 
- [rdkit](https://rdkit.org/)    
- [pytorch](https://pytorch.org/) 
- [dgl](https://www.dgl.ai/)
- [dgllife](https://lifesci.dgl.ai/index.html)

## Adduct graph generation
**1.** Generate adduct ggl-graph of molecules. 
GraphCCS is a model for predicting CCS based on graph convolutional networks, so we need to convert SMILES strings to Adduct Graph. The related method is shown in [`GrapgCCS/dataset.py`](GraphCCS/dataset.py)
    mol = Chem.MolFromSmiles(smi)
    v_ds = edit_adduct_mol(mol, add)
    v_d = fc(mol = v_ds, node_featurizer = node_featurizer, edge_featurizer = None,explicit_hydrogens = True)
*Optionnal args*
- smi : Molecular SMILES,string
- add : Adduct type of molecule,string

**2.** Generate the Graph dataset.

    data = graph_calculation(dataset)
    data_generator = data.DataLoader(data_process_loader_Property_Prediction(data.index.values,  data.Label.values,data), **params)
*Optionnal args*
- dataset : A DataFrame file including SMILES, Adduct Type and experimental CCS values

## Model training
Train the model based on your own training dataset with [train](https://github.com/tingxiecsu/GraphCCS/blob/main/GraphCCS/train.py) function.

    graphccs = Train(train,val,test,**config)
    graphccs.train_()

## Predicting CCS
The CCS prediction of the molecule is obtained by feeding the Adduct Graph into the already trained model with [Model_prediction](https://github.com/tingxiecsu/GraphCCS/blob/main/GraphCCS/train.py#L251) function.

    model_predict = Predict(dataset,model_path,**config)
*Optionnal args*
- dataset : A DataFrame file including SMILES, Adduct Type
- model_path: File path where the model is stored

## Information of maintainers
- zmzhang@csu.edu.cn
- 212307003@csu.edu.cn
