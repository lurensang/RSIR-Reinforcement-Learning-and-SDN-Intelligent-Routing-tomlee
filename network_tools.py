import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G, pos=None, opt=False) -> None:
    # use layout for reproducibility
    if not pos:
        # if not set, use random
        pos = nx.spring_layout(G)
    
    if opt == False:
        # default setting
        options = {
            'with_labels': True,
        }
    else:
        # draw nodes and edges with customized settings
        if isinstance(opt, dict):
            options = opt
        else:
            options = {}
        
        # load color info
        edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]
        node_color_list = [G.nodes[n]['color'] for n in G.nodes()]

        # basic options
        if 'edge_color' not in options:
            options['edge_color'] = edge_color_list
        if 'node_color' not in options:
            options['node_color'] = node_color_list
        if 'with_labels' not in options:
            options['with_labels'] = True
    
    # draw network
    nx.draw_networkx(G, pos, **options)

def create_network_plot(links: dict, pos=None) -> nx.classes.graph.Graph:
    '''
    generate a graph for the system given the links
    '''
    sws = list(links.keys()) # all nodes

    # create graph from links
    G = nx.Graph()
    for i in sws:
        for j in links[i]:
            G.add_edge(i, j)
            set_edge_color(G, i, j, 'black')
    
    for i in G.nodes():
        set_node_color(G, i, 'blue')

    draw_graph(G, pos)

    return G

def set_edge_color(G: nx.classes.graph.Graph, start: int, end: int, color: str) -> None:
    '''
    set color for an edge
    '''
    G[start][end]['color'] = color

def set_node_color(G: nx.classes.graph.Graph, node: int, color: str) -> None:
    '''
    set color for a node
    '''
    G.nodes[node]['color'] = color

