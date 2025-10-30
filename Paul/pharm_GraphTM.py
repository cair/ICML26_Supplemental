from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.Draw import MolsToGridImage
from pharmacophore import Pharmacophore, Draw, View
from chembl_structure_pipeline import standardize_mol as chembl_standardizer
from GraphTsetlinMachine.graphs import Graphs as original_Graphs
from graph import Graphs

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.special import expit
from netgraph import Graph
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_curve, precision_score

from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def mol_from_smiles(smiles, *args, standardizer=chembl_standardizer) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit Mol object

    :param smiles: SMILES string. Required.
    :param standardizer: function to standardize the molecule. Defaults to ChEMBL standardizer. Use None to skip.
    :return: RDKit Mol object
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return None
    if standardizer is None:
        return mol
    return standardizer(mol, check_exclusion=True, sanitize=True)

def condense_features(pharma):
    pharma_arr = np.array(pharma, dtype=object)
    condensed_pharma = []
    atom_index = pharma_arr[:, 1]
    grouped = pd.Series(range(len(atom_index))).groupby(atom_index, sort=False).apply(list).tolist()
        
    for group in grouped:
        features = pharma_arr[group, 0]
        x_coords = pharma_arr[group, 2].astype(float)[0]
        y_coords = pharma_arr[group, 3].astype(float)[0]
        z_coords = pharma_arr[group, 4].astype(float)[0]
        condensed_pharma.append([features, atom_index[group[0]], x_coords, y_coords, z_coords])
    
    return condensed_pharma

def distance_matrix_pharma(condensed_pharma):
    cond_pharma_arr = np.array(condensed_pharma, dtype=object)
    coords = cond_pharma_arr[:, 2:5].astype(float)
    dist_matrix = distance_matrix(coords, coords)
    return dist_matrix

def smiles_to_1DPharma(smiles, pharm_features="default", random_seeds=[1,2,3,4,5]):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    pharm = Pharmacophore(features=pharm_features)
    distance_matrices = []
    for seed in random_seeds:
        ps = AllChem.ETKDGv3()
        ps.randomSeed = seed
        embed_ID = AllChem.EmbedMolecule(mol, ps)
        if embed_ID != -1:
            mol_noH = Chem.RemoveHs(mol)
            pharma = pharm.calc_pharm(mol_noH)
            if len(pharma) > 0:
                condensed_pharma = condense_features(pharma)
                distance_matrices.append(distance_matrix_pharma(condensed_pharma))
                # feature details, edge lengths averaged over conformations and rounded to nearest whole number
                return np.array(condensed_pharma, dtype=object)[:,0:2], np.round(np.average(distance_matrices, axis=0), 0).astype(int)
            else:
                return None
        else:
            return None

def dataset_to_1DPharmaCAIRGraphs(df, smiles_col="SMILES", pharm_features="default", random_seeds=[1,2,3,4,5]):
    
    df['mol'] = df[smiles_col].apply(mol_from_smiles)
    df.dropna(subset=['mol'], inplace=True)
    print("--- Creating pharmacophores ---")
    df['pharmacophore'] = df['SMILES'].apply(smiles_to_1DPharma, pharm_features=pharm_features, random_seeds=random_seeds)
    df.dropna(subset=['pharmacophore'], inplace=True)
    pharma_list = df['pharmacophore'].tolist()
    
    n_mols = len(pharma_list)
    pharm_gen = Pharmacophore(features=pharm_features)
    features = list(eval(pharm_gen.feature_types().split("\n")[1]))

    print("--- Creating CAIR Graphs ---")
    cair_graphs = original_Graphs(
        n_mols,
        symbols = features,
        hypervector_size = 2**len(features),
        hypervector_bits = 2,
        double_hashing=False
    )
    
    for mol_indx, pharamcophore in enumerate(pharma_list):
        condensed_pharma, _ = pharamcophore
        cair_graphs.set_number_of_graph_nodes(mol_indx, len(condensed_pharma))
        
    cair_graphs.prepare_node_configuration()
    for mol_indx, pharamcophore in enumerate(pharma_list):
        condensed_pharma, _ = pharamcophore
        n_edges = len(condensed_pharma) - 1
        
        for feature_point_indx in range(len(condensed_pharma)):
            cair_graphs.add_graph_node(
                mol_indx,
                feature_point_indx,
                n_edges
            )
    
    cair_graphs.prepare_edge_configuration()
    
    for mol_indx, pharamcophore in enumerate(pharma_list):
        condensed_pharma, dist_matrix = pharamcophore
        for i in range(len(dist_matrix)):
            for j in range(len(dist_matrix)):
                if i != j:
                    edge_length = dist_matrix[i][j]
                    cair_graphs.add_graph_node_edge(
                        mol_indx,
                        i,
                        j,
                        edge_length
                    )
               
            for feature_indx, feature in enumerate(condensed_pharma[i][0]):    
                cair_graphs.add_graph_node_property(
                    mol_indx,
                    i,
                    feature
                )          
    cair_graphs.encode()
    return cair_graphs, df

# visualization
def pharma1D_nx_graph(condensed_pharma, dist_matrix):
    G = nx.Graph()
    for feature_point_indx in range(len(condensed_pharma)):
        feature_dict = {}
        for feature_indx, feature in enumerate(condensed_pharma[feature_point_indx][0]):
            feature_dict[feature_indx] = feature        
        G.add_nodes_from([(feature_point_indx, feature_dict)])
    for i in range(len(dist_matrix)):
        for j in range(i+1, len(dist_matrix)):
            G.add_edge(i, j, weight=dist_matrix[i][j])
    return G


# PRC_AUC
def prc_auc_score(Y, Y_pred):
        precision, recall, _ = precision_recall_curve(Y, Y_pred)
        return auc(recall, precision)

# Precision_PPV, Precision_NPV
def ppv_npv_score(Y, Y_pred):
        fpr, tpr, proba = roc_curve(Y, Y_pred)
        optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), proba)),\
                                      key=lambda i: i[0], reverse=True)[0][1]
        
        hard_Y_pred = [1 if p > optimal_proba_cutoff else 0 for p in Y_pred]

        return precision_score(Y, hard_Y_pred, pos_label=1, zero_division=0), precision_score(Y, hard_Y_pred, pos_label=0, zero_division=0)
    
PHARM_FEATURES = "rdkit"
PROPERTY_LABEL = "label"
df = pd.read_csv("./datasets/opioids/MOR_cutoff6.csv")

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[PROPERTY_LABEL])

X_graph_train, clean_df_train = dataset_to_1DPharmaCAIRGraphs(df=df_train, pharm_features=PHARM_FEATURES)
X_graph_test, clean_df_test = dataset_to_1DPharmaCAIRGraphs(df=df_test, pharm_features=PHARM_FEATURES)


N_CLAUSES = 1000
T = 500*10
s = 5.0
DEPTH = 2
MP_SIZE = 10000

N_EPOCHS = 5


graph_TM = MultiClassGraphTsetlinMachine(
    number_of_clauses=N_CLAUSES,
    T=T,
    s=s,
    depth=DEPTH,
    message_size=MP_SIZE,
    message_bits=2,
    number_of_state_bits=15
)

print("--- Training Graph Tsetlin Machine ---")
for epoch in range(N_EPOCHS):
    
    graph_TM.fit(X_graph_train, clean_df_train[PROPERTY_LABEL].values, epochs=1, incremental=True)
    
    Y_train_css = graph_TM.score(X_graph_train)[:,1]
    Y_train_pred_prob = expit(Y_train_css/graph_TM.T)
    
    Y_test_css = graph_TM.score(X_graph_test)[:,1]
    Y_test_pred_prob = expit(Y_test_css/graph_TM.T)
    
    train_prc_auc = prc_auc_score(clean_df_train[PROPERTY_LABEL].values, Y_train_pred_prob)
    test_prc_auc = prc_auc_score(clean_df_test[PROPERTY_LABEL].values, Y_test_pred_prob)
    
    train_ppv, train_npv = ppv_npv_score(clean_df_train[PROPERTY_LABEL].values, Y_train_pred_prob)
    test_ppv, test_npv = ppv_npv_score(clean_df_test[PROPERTY_LABEL].values, Y_test_pred_prob)
    
    print(f"Epoch {epoch+1}/{N_EPOCHS} - Train PRC AUC: {train_prc_auc:.4f}, Test PRC AUC: {test_prc_auc:.4f}, Train PPV: {train_ppv:.4f}, Test PPV: {test_ppv:.4f}, Train NPV: {train_npv:.4f}, Test NPV: {test_npv:.4f}")

# nx_graph = pharma_nx_graph(condensed_pharma, dist_matrix)
# print(nx_graph.nodes(data=True))
# print(nx_graph.edges(data=True))

# # color network by feature type
# nx.draw(nx_graph, with_labels=True, )