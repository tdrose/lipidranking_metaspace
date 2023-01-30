from itertools import product
from typing import List
import re
import numpy as np
import pandas as pd
import scanpy as sc
import ast
import networkx as nx


def flatten(l: List):
    return [item for sublist in l for item in sublist]


def list_product(l1: List, l2: List):
    return list(product(l1, l2))


def match_lipid(x, pattern=r'^PC') -> bool:
    try:
        return bool(re.search(pattern, x))
    except:
        return False


def match_lipids(x, pattern=r'^PC'):
    if x is not np.nan:
        if len(x) > 1:
            try:
                return match_lipid(x[0], pattern) or match_lipid(x[1], pattern)
            except:
                return False
        else:
            return match_lipid(x[0], pattern)
    else:
        return False


def get_all_conditions(adata, col):
    return list(set(adata.obs[col]))


def get_condition_matrices(adata, col, conditions):
    out_dict = {}
    for cond in conditions:
        out_dict[cond] = adata.X[adata.obs[col] == cond, :]
    return out_dict


def log10_condition_matrices(cond_dict, offset=0):
    return {k: np.log10(v+offset) for k, v in cond_dict.items()}


def mean_condition_matrices(cond_dict):
    return {k: np.nanmean(v, axis=0) for k, v in cond_dict.items()}


def logfc_condition_matrices(cond_dict, c1, c2):
    return cond_dict[c1] - cond_dict[c2]


def nudge(pos, x_shift, y_shift):
    return {n: (x + x_shift, y + (y_shift * np.random.choice([-1, 1]))) for n, (x, y) in pos.items()}


def transform_annotations_to_list(adata: sc.AnnData, col: str = 'moleculeNames'):
    app_list = []
    for i in adata.var.index:
        app_list.append(ast.literal_eval(adata.var.loc[i, col]))

    adata.var[col] = pd.Series(app_list, index=adata.var.index)


def get_component(net, position=-1):
    cc = list(nx.connected_components(net))
    order = np.argsort([len(x) for x in cc])

    return nx.induced_subgraph(net, cc[order[position]])
