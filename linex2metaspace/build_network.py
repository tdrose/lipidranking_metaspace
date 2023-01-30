from typing import List, Dict, Union
import random
from collections import defaultdict

import networkx as nx
import pandas as pd
import numpy as np
import linex2 as lx2

from .utils import list_product
from .linex2_processing import get_lx2_ref_lip_dict, get_lx2_ref_lips


def parse_annotation_series(x: pd.Series,
                            ref_lips: dict,
                            verbose=True) -> pd.Series:
    parsed_list = []
    for i, v in x.items():
        tmp_list = []
        for lip in v:
            pl = None

            try:
                pl = lx2.lipid_parser(lip, is_ll_compatible=True, reference_lipids=ref_lips,
                                      org_name=str(i), convert_plasmogen=True,
                                      force_ether_db_pos=True)  # Assign ion as name
                pl.set_converted_to_mol_spec(converted=True)
            except:
                if verbose:
                    print(f'Could not parse: \'{lip}\'')
            if pl is not None:
                tmp_list.append(pl)
        parsed_list.append(tmp_list)
    return pd.Series(parsed_list, index=x.index)


def annotations_parsed_lipids(x: pd.Series) -> List:
    return list(x[x.apply(lambda y: len(y)) > 0].index)


def unique_sum_species(x: pd.Series) -> pd.Series:
    return x.apply(lambda z: list(set([y.sum_species_str() for y in z])))


def sample_sum_species(x: pd.Series):
    return x.apply(lambda y: random.choice(y))


def select_lipids_from_sample(all_lipids: pd.Series,
                              sum_sample: pd.Series) -> pd.Series:
    out_l = []
    for i in all_lipids.index:
        out_l.append([lip for lip in all_lipids[i] if lip.sum_species_str() == sum_sample[i]])
    return pd.Series(out_l, index=all_lipids.index)


def make_lipid_dict(x: pd.Series) -> Dict:
    out_dict = dict()
    for ll in x:
        for l in ll:
            if l.get_lipid_class() not in out_dict.keys():
                out_dict[l.get_lipid_class()] = [l]
            else:
                out_dict[l.get_lipid_class()].append(l)
    return out_dict


def unite_networks(x: List[nx.MultiGraph]) -> nx.MultiGraph:
    g = nx.MultiGraph()

    for graph in x:
        for n, d in graph.nodes(data=True):
            g.add_node(n, **d)
        for u, v, d in graph.edges(data=True):
            g.add_edge(u, v, **d)

    return g


def bootstrap_networks(unique_species: pd.Series,
                       parsed_lipids: pd.Series,
                       n: int,
                       lx2_class_reacs: List,
                       lx2_reference_lipids: List,
                       return_composed=False,
                       verbose=False) -> Union[nx.Graph,
                                                       nx.MultiGraph,
                                                       List[Union[nx.Graph,
                                                                  nx.MultiGraph]]]:
    net_list = []
    for i in range(n):
        print(str(i+1), '/', n)
        samp = sample_sum_species(unique_species)
        bootstrapped_lipids = select_lipids_from_sample(parsed_lipids, samp)

        gln = lx2.GenerateLipidNetwork(
            reference_lipids=lx2_reference_lipids,
            lipids=make_lipid_dict(bootstrapped_lipids),
            class_reactions=lx2_class_reacs,
            fa_reactions=[],
            ether_conversions=False
        )
        tmp = gln.native_network(filter_duplicates=False, multi=True, verbose=verbose)
        for u, v, t in tmp.edges:
            tmp[u][v][t]['bootstrap'] = i

        net_list.append(tmp)

    if return_composed:
        return unite_networks(net_list)

    return net_list


def lipid_ion_graph(g: Union[nx.Graph, nx.MultiGraph],
                    sum_species: pd.Series) -> Union[nx.Graph, nx.MultiGraph]:
    gn = g.copy()

    for ion, ll in sum_species.items():
        for lip in ll:
            gn.add_edge(ion, lip, ion_lipid_edge=True)

    for n in gn.nodes():
        if 'node_molecule_type' in gn.nodes[n].keys():
            gn.nodes[n]['color'] = 'lightblue'
        else:
            gn.nodes[n]['color'] = 'red'
            gn.nodes[n]['node_molecule_type'] = 'ion'

    return gn


def ion_weight_graph(g: nx.MultiGraph,
                     sum_species: pd.Series,
                     bootstraps: int,
                     parsed_lipids: pd.Series = None,
                     feature_similarity: pd.DataFrame = None) -> nx.Graph:
    gn = g.copy()

    # Add dataname to nodes
    for ion, ll in sum_species.items():
        for lip in ll:
            if lip in gn.nodes.keys():
                if 'dataname' in gn.nodes[lip].keys():
                    gn.nodes[lip]['dataname'].append(ion)
                else:
                    gn.nodes[lip]['dataname'] = [ion]

    # Make ion graph
    ig = nx.MultiGraph()
    for u, v, d in gn.edges(data=True):
        for d1 in gn.nodes[u]['dataname']:
            for d2 in gn.nodes[v]['dataname']:
                srt = sorted([d1, d2])
                ig.add_edge(srt[0], srt[1], **d)

    # Weight annotation based on edge count
    node_weight_dict = {}  # Key is lipid, value is dict of weight and dataname

    for n in gn.nodes:  # Loop over all lipids
        node_weight_dict[n] = {'weight': 0, 'dataname': gn.nodes[n]['dataname']}
        bt_dict = defaultdict(int)  #
        tmp_bt_dict = defaultdict(int)  # Key: Bootstrap, val: counter
        # Loop over all neighbors
        #  for val in gn[n].values():
        for olip in gn[n].keys():
            val = gn[n][olip]
            for x in set(val.keys()):
                # Instead:
                tmp_bt_dict[val[x]['bootstrap']] += 1
                # if feature_similarity is None:
                #       # Unweighted: every neighbor is counted equally
                #       tmp_bt_dict[val[x]['bootstrap']] += 1
                # else:
                #     # Weighted: neighbors contribute with their similarity
                #     weights = [feature_similarity.loc[d1, d2] for d1, d2 in list_product(gn.nodes[n]['dataname'],
                #                                                                          gn.nodes[olip]['dataname'])]
                #     print(weights)
                #     tmp_bt_dict[val[x]['bootstrap']] += np.mean(weights)

                # bt_dict[x] += 1
            for k1, v1 in tmp_bt_dict.items():
                # k1 bootstrap
                # v1 counter
                if v1 > 0:
                    # bt_dict[k1] += 1
                    # Instead:
                    if feature_similarity is None:
                        # Unweighted: every neighbor is counted equally
                        bt_dict[k1] += 1
                    else:
                        # Weighted: neighbors contribute with their similarity
                        weights = [feature_similarity.loc[d1, d2] for d1, d2 in
                                   list_product(gn.nodes[n]['dataname'],
                                                gn.nodes[olip]['dataname'])]
                        #print(weights)
                        bt_dict[k1] += np.mean(weights)

        if(len(bt_dict)) > 0:
            node_weight_dict[n]['weight'] = np.mean(list(bt_dict.values()))

    ion_weight_dict = {}
    for k, v in node_weight_dict.items():
        for ion in v['dataname']:
            if ion in ion_weight_dict.keys():
                ion_weight_dict[ion][k] = v['weight']
            else:
                ion_weight_dict[ion] = {k: v['weight']}

    #print(ion_weight_dict)

    # Create final simple ion_graph
    ign = nx.Graph()
    for ind in sum_species.index:
        if parsed_lipids is None:
            ign.add_node(ind,
                         sum_species=pd.Series(ion_weight_dict[ind]).sort_values(ascending=False)
                         )
        else:
            ign.add_node(ind,
                         sum_species=pd.Series(ion_weight_dict[ind]).sort_values(ascending=False),
                         parsed_lipids=parsed_lipids[ind]
                         )

    test_l = []
    for u, v, d in ig.edges(data=True):
        srt = sorted([u, v])
        if tuple(sorted((srt[0], srt[1]))) in test_l:
            ign[srt[0]][srt[1]]['enzyme_id'] += d['enzyme_id'].split(';')
            ign[srt[0]][srt[1]]['enzyme_id'] = list(set(ign[srt[0]][srt[1]]['enzyme_id']))

            ign[srt[0]][srt[1]]['enzyme_gene_name'] += d['enzyme_gene_name'].split(', ')
            ign[srt[0]][srt[1]]['enzyme_gene_name'] = list(set(ign[srt[0]][srt[1]]['enzyme_gene_name']))

            ign[srt[0]][srt[1]]['enzyme_uniprot'] += d['enzyme_uniprot'].split(', ')
            ign[srt[0]][srt[1]]['enzyme_uniprot'] = list(set(ign[srt[0]][srt[1]]['enzyme_uniprot']))
        else:
            test_l.append(tuple(sorted((srt[0], srt[1]))))

            # weighting
            ign.add_edge(srt[0], srt[1],
                         weight=len(set([x['bootstrap'] for x in dict(ig[u][v]).values()])) / bootstraps,
                         enzyme_id=d['enzyme_id'].split(';'),
                         enzyme_gene_name=d['enzyme_gene_name'].split(', '),
                         enzyme_uniprot=d['enzyme_uniprot'].split(', ')
                         )

    ign.remove_edges_from(nx.selfloop_edges(ign))

    return ign


def make_lipid_networks(ann: pd.DataFrame,
                        class_reacs,
                        lipid_col: str = 'moleculeNames',
                        bootstraps=30,
                        verbose=True):

    parsed_lipids = parse_annotation_series(ann[lipid_col], get_lx2_ref_lip_dict(), verbose=verbose)
    keep_annotations = annotations_parsed_lipids(parsed_lipids)
    parsed_annotations = ann.copy()
    parsed_annotations['parsed_lipids'] = parsed_lipids
    parsed_annotations = parsed_annotations.loc[keep_annotations, :]

    g = bootstrap_networks(
        unique_sum_species(parsed_annotations['parsed_lipids']),
        parsed_annotations['parsed_lipids'],
        n=bootstraps,
        lx2_class_reacs=class_reacs,
        lx2_reference_lipids=get_lx2_ref_lips(),
        return_composed=True,
        verbose=verbose
    )

    ig = ion_weight_graph(g, unique_sum_species(parsed_annotations['parsed_lipids']), bootstraps=bootstraps)

    return parsed_annotations, g, ig