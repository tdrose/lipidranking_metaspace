import pandas as pd
import numpy as np
from typing import Dict
import linex2 as lx2
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

from .utils import match_lipid


def lipid_bubble_plot(likelilipids: pd.Series,
                      condition1: str,
                      condition2: str,
                      reference_lipids,
                      condition_data_dict: Dict[str, np.array],
                      parsed_lipids: pd.Series,
                      regex_pattern: str = r'\([0-9]+',
                      ):
    """

    Args:

    Returns:
    
    """
    position_dict = {}

    for i in tqdm(range(len(likelilipids))):
        if likelilipids[i] is not np.nan:
            if match_lipid(likelilipids[i][0], pattern=regex_pattern):

                # Only use this to get the class
                tmp1 = lx2.lipid_parser(likelilipids[i][0], reference_lipids=reference_lipids)
                for ps in parsed_lipids[likelilipids.index[i]]:
                    if tmp1.get_lipid_class() == ps.get_lipid_class():
                        tmp = ps
                        break

                tk = (tmp.sum_length(), tmp.sum_dbs())
                if tk in position_dict.keys():
                    position_dict[tk].append(i)
                else:
                    position_dict[tk] = [i]
            elif match_lipid(likelilipids[i][1], pattern=regex_pattern):

                # Only use this to get the class
                tmp1 = lx2.lipid_parser(likelilipids[i][1], reference_lipids=reference_lipids)
                for ps in parsed_lipids[likelilipids.index[i]]:
                    if tmp1.get_lipid_class() == ps.get_lipid_class():
                        tmp = ps
                        break

                tk = (tmp.sum_length(), tmp.sum_dbs())
                if tk in position_dict.keys():
                    position_dict[tk].append(i)
                else:
                    position_dict[tk] = [i]

    mean_dict_c1 = {}
    for k, v in position_dict.items():
        mean_dict_c1[k] = condition_data_dict[condition1][:, v].sum(axis=1).mean()
    mean_dict_c2 = {}
    for k, v in position_dict.items():
        mean_dict_c2[k] = condition_data_dict[condition2][:, v].sum(axis=1).mean()

    df1 = pd.DataFrame(mean_dict_c1.items()).rename(columns={0: 'Pos', 1: 'C1'})
    df1['C'] = df1['Pos'].apply(lambda x: x[0])
    df1['DB'] = df1['Pos'].apply(lambda x: x[1])
    df2 = pd.DataFrame(mean_dict_c2.items()).rename(columns={0: 'Pos', 1: 'C2'})
    df2['C'] = df2['Pos'].apply(lambda x: x[0])
    df2['DB'] = df2['Pos'].apply(lambda x: x[1])
    final_df = pd.concat([df1.set_index(['C', 'DB']).drop(columns=['Pos']),
                          df2.set_index(['C', 'DB']).drop(columns=['Pos'])], axis=1).reset_index()
    final_df['LogFC'] = final_df['C1'] - final_df['C2']
    final_df['|LogFC|'] = abs(final_df['C1'] - final_df['C2'])

    return final_df


def plot_ion_network(net, plot_edge_labels=False, pos=None, k=.15, return_pos=False):
    """

    Args:

    Returns:
    
    """
    if pos is None:
        pos = nx.spring_layout(net, k=k)
    nx.draw_networkx_nodes(net, pos=pos, node_size=12, )
    nx.draw_networkx_edges(net, pos=pos, width=[d['weight'] for u, v, d in net.edges(data=True)])
    if plot_edge_labels:
        nx.draw_networkx_edge_labels(net,
                                     pos=pos,
                                     edge_labels={(u, v): round(d['weight'], 2) for u, v, d in net.edges(data=True)},
                                     font_size=4)
    nx.draw_networkx_labels(net, pos=pos,
                            labels={k: ", ".join(v['sum_species'].index) for k, v in
                                    dict(net.nodes(data=True)).items()},
                            font_size=4)
    plt.show()

    if return_pos:
        return pos
