import numpy as np
from scipy.sparse import csr_matrix
from time import time
import argparse

import pandas as pd

from graphtm_exp.graph import Graphs
from graphtm_exp.benchmark import Benchmark

import pickle

import os


def calc_graph_info(df, state_headers, action_headers):
    df['final_state'] = '1:9 2:9 3:9 4:9 5:9'
    calc_state_headers = state_headers[:-1]+['final_state']
    for ind, row in df.iterrows():
        allstates = row[calc_state_headers].tolist()
        uniquestates = []
        node_edge_cnts =[]
        for st in allstates:
            if st not in uniquestates:
                uniquestates.append(st)
        edgedata = row[action_headers].tolist()
        outedgecnts= {nd:0 for nd in range(len(uniquestates))}
        inedgecnts= {nd:0 for nd in range(len(uniquestates))}
        df.loc[ind, 'uniquestatescnt'] = int(len(uniquestates))
        df.loc[ind, 'uniquestates'] = str(uniquestates)
        edges = []
        for ed in range(len(edgedata)):
            fromstate = allstates[ed]
            tostate = allstates[ed+1]
            from_node_id = uniquestates.index(fromstate)
            to_node_id = uniquestates.index(tostate)
            outedgecnts[from_node_id]+= 1
            inedgecnts[to_node_id]+= 1
            edges.append((from_node_id, to_node_id))
        df.loc[ind, 'calculatededges'] = str(edges)
        df.loc[ind, 'outedges'] = str(outedgecnts)
        df.loc[ind, 'inedges'] = str(inedgecnts)
        
    df['uniquestatescnt']=df['uniquestatescnt'].astype(int)


# Read the CSV file
def create_train_test_graphs(tm_parameters, headers, filename, devfilename, opfilename):
    raw_data_inp= pd.read_csv(filename, sep='\t', names=headers)
    raw_data_dev= pd.read_csv(devfilename, sep='\t', names=headers)
    raw_data_inp =raw_data_inp.dropna(ignore_index=True)
    raw_data_dev=raw_data_dev.dropna(ignore_index=True)

    '''Remove following for 5-utterance
    headers_3utt =  ['rowid', 'state1', 'action1','state2', 'action2', 'state3', 'action3', 'state4']
    dropheaders = [h for h in headers if h not in headers_3utt]
    raw_data_inp = raw_data_inp.drop(columns=dropheaders)
    raw_data_dev = raw_data_dev.drop(columns=dropheaders)
    headers = headers_3utt
    Remove for 5-utterance'''
    # Read the CSV file

    #Process data from files for graph input
    max_number_of_nodes = 6 ##Max number of states per example
    action_headers = [h for h in headers if 'action' in h]
    state_headers = [h for h in headers if 'state' in h]

    number_of_classes=6 ##5 possible figures and Nothing
    number_of_examples_train=len(raw_data_inp)
    number_of_examples_dev=len(raw_data_dev)

    qpos_possibilities=['1','2','3','4','5'] ##Query: which image is in postion x? e.g. x=1

    calc_graph_info(raw_data_inp, state_headers, action_headers)
    calc_graph_info(raw_data_dev, state_headers, action_headers)

    print(raw_data_inp)

    #Process data from files for graph input

    print("Creating training data")

    ##Make Symbols :: combination of position(1-5) and imagenumbers(0-4 and _)
    symbols = []
    for pos in range(1,6):
        for img in range(0,5):
            symbols.append('%d:%d'%(pos,img))
        symbols.append('%d:_'%pos)
        symbols.append('%d:9'%pos)

    ## Create train graphs:: holder

    graphs_train = Graphs(
        number_of_examples_train,
        symbols=symbols,
        hypervector_size=tm_parameters['hypervector_size'],
        hypervector_bits=tm_parameters['hypervector_bits'],
        double_hashing = tm_parameters['double_hashing']
    )

    # Create train graphs:: nodes
    for graph_id in range(number_of_examples_train):
        graphs_train.set_number_of_graph_nodes(graph_id, raw_data_inp.loc[graph_id, 'uniquestatescnt'])

    graphs_train.prepare_node_configuration()

    for graph_id in range(number_of_examples_train):
        outedgecnts = eval(raw_data_inp.loc[graph_id, 'outedges'])
        nodenames = eval(raw_data_inp.loc[graph_id, 'uniquestates'])
        for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
            number_of_edges = outedgecnts[node_id]
            graphs_train.add_graph_node(graph_id, node_id, number_of_edges)
            for prop in nodenames[node_id].split(' '):
                graphs_train.add_graph_node_property(graph_id, node_id, prop)



    # Create train graphs:: edges
    graphs_train.prepare_edge_configuration()


    for graph_id in range(number_of_examples_train):
        alledges = eval(raw_data_inp.loc[graph_id, 'calculatededges'])
        edgedata = raw_data_inp.loc[graph_id, action_headers].tolist()
        edgenum = 0
        for edge in alledges:
            from_node_id = int(edge[0])
            to_node_id = int(edge[1])
            graphs_train.add_graph_node_edge(graph_id, from_node_id, to_node_id, edgedata[edgenum])
            edgenum+= 1
        

    # Create train graphs:: output
    ##Which figure is in 1st position? 
    #Alt :: 2nd/3rd/4th/5th position
    qpos = '1'
    Y_train = np.empty(number_of_examples_train, dtype=np.uint32)
    for graph_id in range(number_of_examples_train):
        ans = raw_data_inp.loc[graph_id, state_headers[-1]].split(' ')
        for answer in ans:
            if qpos+':' in answer: ##Question pertains to x-th position
                ansasint = answer.replace(qpos+':','')
                if ansasint!='_':
                    ansasint = int(ansasint)
                else:
                    ansasint= 6
                Y_train[graph_id] =  ansasint
                break
            else:
                Y_train[graph_id] = 6
                
    graphs_train.encode()
    print("Training data produced")


    print("Creating Dev data")

    # Create dev graphs:: holder

    graphs_dev = Graphs(len(raw_data_dev), init_with=graphs_train)
    # Create test graphs:: nodes
    for graph_id in range(number_of_examples_dev):
        graphs_dev.set_number_of_graph_nodes(graph_id, raw_data_dev.loc[graph_id, 'uniquestatescnt'])

    graphs_dev.prepare_node_configuration()

    for graph_id in range(number_of_examples_dev):
        outedgecnts = eval(raw_data_dev.loc[graph_id, 'outedges'])
        nodenames = eval(raw_data_dev.loc[graph_id, 'uniquestates'])
        for node_id in range(graphs_dev.number_of_graph_nodes[graph_id]):
            number_of_edges = outedgecnts[node_id]
            graphs_dev.add_graph_node(graph_id, node_id, number_of_edges)
            for prop in nodenames[node_id].split(' '):
                graphs_dev.add_graph_node_property(graph_id, node_id, prop)


    # Create dev graphs:: edges
    graphs_dev.prepare_edge_configuration()


    for graph_id in range(number_of_examples_dev):
        alledges = eval(raw_data_dev.loc[graph_id, 'calculatededges'])
        edgedata = raw_data_dev.loc[graph_id, action_headers].tolist()
        edgenum = 0
        for edge in alledges:
            from_node_id = int(edge[0])
            to_node_id = int(edge[1])
            graphs_dev.add_graph_node_edge(graph_id, from_node_id, to_node_id, edgedata[edgenum])
            edgenum+= 1
        

    # Create dev graphs:: output
    Y_dev = np.empty(number_of_examples_dev, dtype=np.uint32)
    for graph_id in range(number_of_examples_dev):
        ans = raw_data_dev.loc[graph_id, state_headers[-1]].split(' ')
        for answer in ans:
            if qpos+':' in answer: 
                ansasint = answer.replace(qpos+':','')
                if ansasint!='_':
                    ansasint = int(ansasint)
                else:
                    ansasint= 6
                Y_dev[graph_id] =  ansasint
                break
            else:
                Y_dev[graph_id] = 6

    graphs_dev.encode()
    print("-Dev data complete")

    X_train = raw_data_inp
    X_test = raw_data_dev
    return (graphs_train,graphs_dev, symbols)

if __name__ == "__main__":
    #noise = 0.05


    # Define the relative path (e.g., a folder named 'data' inside the current directory)
    relative_path = "ActionCoreference/data"

    # Get the absolute path
    abs_path = os.path.join(os.getcwd(), relative_path)

    # List all files in that directory
    files = os.listdir(abs_path)

    print("Files in relative path:", files)

    filename = 'ActionCoreference/data/tangrams-train.tsv'
    devfilename = 'ActionCoreference/data/tangrams-dev.tsv'
    opfilename = 'tangrams_3u.txt'

    headers = ['rowid', 'state1', 'action1','state2', 'action2', 'state3', 'action3', 'state4', 'action4', 'state5', 'action5', 'state6']


    ## TM Paramters
    tm_parameters ={'epochs':100,'number_of_clauses':800, 'T':9000,'s':0.05,
                     'number_of_state_bits':8, 'depth':6,'hypervector_size':256, 'hypervector_bits':4,
                      'message_size':256, 'message_bits':2,'double_hashing':False, 'noise':0.01, 
                      'max_sequence_length':1000, 'max_included_literals':40}
    ## TM Paramters
    (graphs_train, graphs_dev, symbols) = create_train_test_graphs(tm_parameters,headers, filename, devfilename,opfilename)
    with open('ActionCoreference/action_coref_binarized_5utt.pkl', 'rb') as f:
        binarized_data =pickle.load(f)

    X_train = binarized_data['X_train']
    Y_train = binarized_data['Y_train']
    X_test = binarized_data['X_test']
    Y_test = binarized_data['Y_test']



    save_dir = "./temp"

    gtm_params = {
        "number_of_clauses": tm_parameters['number_of_clauses'],
        "T": tm_parameters['T'],
        "s": tm_parameters['s'],
        "message_size": tm_parameters['message_size'],
        "message_bits": tm_parameters['message_bits'],
        "double_hashing": tm_parameters['double_hashing'],
        "depth": tm_parameters['depth'],
        # "grid": (16 * 13, 1, 1),
        # "block": (128, 1, 1),
    }

    vantm_params = {
        "number_of_clauses": tm_parameters['number_of_clauses'],
        "T": tm_parameters['T'],
        "s": tm_parameters['s'],
        "platform": "GPU",
        "weighted_clauses": True,
    }

    cotm_params = {
        "number_of_clauses": tm_parameters['number_of_clauses'],
        "T": tm_parameters['T'],
        "s": tm_parameters['s'],
        "dim": (X_train.shape[1], 1, 1),
        "patch_dim": (X_train.shape[1], 1),
        # "grid": (16 * 13, 1, 1),
        # "block": (128, 1, 1),
    }

    xgb_params = {}

    bm = Benchmark(
        X=X_train,
        Y=Y_train,
        graphs=graphs_train,
        save_dir=save_dir,
        name="actioncoref_5utt",
        #gtm_args=gtm_params,
        #xgb_args=xgb_params,
        #vanilla_tm_args=vantm_params,
        cotm_args=cotm_params,
        X_test=X_test,
        Y_test=Y_test,
        graphs_test=graphs_dev,
        epochs=10,
    )
    bm.run()