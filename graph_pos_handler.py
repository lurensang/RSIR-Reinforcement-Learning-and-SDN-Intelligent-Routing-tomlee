import numpy as np
import networkx as nx
 
def save_graph(G: nx.classes.graph.Graph, graph_name: str) -> None:
    '''
    save graph edges to file

    do not include extentions in graph_name
    '''
    graph_name_full = graph_name + '.gml'
    nx.write_gml(G, graph_name_full)

def load_graph(graph_name: str) -> nx.classes.graph.Graph:
    '''
    load graph edges info from file

    do not include extentions in graph_name
    '''
    graph_name_full = graph_name + '.edgelist'
    G = nx.read_gml(graph_name_full, destringizer=int)
    return G

def save_pos(file_name: str, pos:dict) -> None:
    '''
    save position info to npy file
    '''
    np.save(file_name, pos)

def load_pos(file_name: str) -> dict:
    '''
    load position info from npy file and convert to dict
    '''
    pos_np = np.load(file_name, allow_pickle=True)
    pos = pos_np.reshape(1,)[0]
    return pos