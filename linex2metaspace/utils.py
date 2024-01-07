from itertools import product
from typing import List, Iterable, Dict, Union
import re
import numpy as np
import pandas as pd
import scanpy as sc
import ast
import networkx as nx
from anndata import AnnData


def flatten(l: List):
    return [item for sublist in l for item in sublist]


def list_product(l1: List, l2: List):
    return list(product(l1, l2))


def match_lipid(x, pattern=r'^PC') -> bool:
    try:
        return bool(re.search(pattern, x))
    except:
        return False


def match_lipids(x: Iterable, pattern: str=r'^PC'):
    """
    Match lipid string representations to a specific class.

    Args:
        x: Iterable of lipid names. Matches the first two in the list to a class pattern.
        pattern: Regex string to match lipid names against.

    Returns:
        True if the first or second lipid names in the list match the pattern.
    
    """
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


def get_all_conditions(adata: AnnData, col: str) -> List[str]:
    """
    Extracts all conditions from an ``AnnData.obs`` table. 

    Args:
        adata: An AnnData object
        col: Column name of the column containing conditions of interest.
    
    Returns:
        List of conditions
    """
    return list(set(adata.obs[col]))


def get_condition_matrices(adata: AnnData, col: str, 
                           conditions: List[str]) -> Dict[str, np.ndarray]:
    
    """
    Extract data submatrix  containing all samples per condition.

    Args:
        adata: an AnnData object
        col: Column name of the column containing conditions of interest.
        condition: List of all possible conditions in ``adata.obs[col]`` as 
        returned by ``get_all_conditions``
    
    Returns:
        Dictionary with condition as key and submatrix extracted from  ``adata.X`` as value.
    """
    out_dict = {}
    for cond in conditions:
        out_dict[cond] = adata.X[adata.obs[col] == cond, :]
    return out_dict


def log10_condition_matrices(cond_dict: Dict[str, np.ndarray], 
                             offset: float=0) -> Dict[str, np.ndarray]:
    """
    Log10 transform a dict of datamatrices.

    Args:
        cond_dict: Dict with data matrices as values
        offset: value for pseudo log: (log(x + offset))

    Returns:
        Log transformed dict of data matrices

    """
    return {k: np.log10(v+offset) for k, v in cond_dict.items()}


def mean_condition_matrices(cond_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate mean value per feature for a dict of data matrices as 
    returned by  ``get_condition_matrices`` function.

    Args:
        cond_dict: Dict with data matrices as values

    Returns:

    """
    return {k: np.nanmean(v, axis=0) for k, v in cond_dict.items()}


def logfc_condition_matrices(cond_dict: Dict[str, np.ndarray], c1: str, c2: str) -> np.ndarray:
    """
    Calculate the log fold change between the mean value of two conditions per feature.
    Data has to be log transformed before.

    Args:
        cond_dict: Dict with data matrices as values
        c1: Name of a condition for comparison.
        c2: Name of another condition for comparison.

    Returns:
        Array of log-FC values for each feature between values of condition1 and condition2.
    """
    return cond_dict[c1] - cond_dict[c2]


def nudge(pos, x_shift, y_shift):
    """
    Random nudging of positions returned by any ``networkx`` layouting 
        to reduce overlaps in labels.

    Args:
        pos: List of tuples containing x and y positions for each node of a network
        x_shift: each x position is shifted by this value
        y_shift: each y position is shifted randomly between -y_shift and y_shift 
            in the y direction. 

    Returns:
        Modified positions
    """
    return {n: (x + x_shift, y + (y_shift * np.random.choice([-1, 1]))) for n, 
            (x, y) in pos.items()}


def transform_annotations_to_list(adata: AnnData, col: str = 'moleculeNames') -> AnnData:
    """

    When loading an AnnData objeect from h5ad, columns that contain lists are saved as strings.
    This function converts them back to lists.

    Args:
        adata: An AnnData object.
        col: Name of the column in ``adata.var`` which has to be transformed.

    Returns:
        Updated AnnData object with one ``adata.var`` columns transformed using 
        the ``ast.literal_eval`` function.

    """
    app_list = []
    for i in adata.var.index:
        app_list.append(ast.literal_eval(adata.var.loc[i, col])) # type: ignore

    adata.var[col] = pd.Series(app_list, index=adata.var.index)


def get_component(net: Union[nx.Graph, nx.MultiGraph], position=-1) -> Union[nx.Graph, 
                                                                             nx.MultiGraph]:
    """
    Get a component of a network.

    Args:
        net: A networkx graph
        position: Position of the component in a sorted list of components by node size.
            Detault is -1, which will return the largest component

    Returns:
        Induced subgraph of all nodes that are part of the component. 
    
    """
    cc = list(nx.connected_components(net))
    order = np.argsort(np.array([len(x) for x in cc]))

    return nx.induced_subgraph(net, cc[order[position]])

import pandas as pd
import numpy as np

def candidate_selection(ann: pd.DataFrame, ion_network: nx.Graph, cutoff: int=1) -> pd.DataFrame:
    """
    Get the most likely candidate lipids per annotation from an ion network.

    Args:
        ann: Annotation table (as returned by ``make_lipid_networks`` function). 
            New information will be added as columns to a copy of this table
        ion_network: Ion (annotation) network from which lipidr anking iformation is extracted 
            (as returned by ``make_lipid_networks`` function)
        cutoff: Number of most likely candidate lipids to be extracted per annotation.
            We recommend to use the default value of 1 (as evaluated for the pbulication).

    Returns:
        A extended copy of the ``ann`` annotation table with the following new columns:
        
        * `is_ranked` stores whether a ranking could be performed for this annotation. 
          E.g. if none of the lipid candidates had any connections, ranking was not performed 
          for this lipid.
        * `utilized_sum_species` provides a list of all lipids that were used for the ranking.
        * `ranked_lipids` contains all lipids with at least one connection during 
          bootstrapping ranked in descending order.
        * `candidates` contains a list of lipids as candidates molecules per annotation. 
          The number of lipids is decided by the `cutoff` parameter 
          (number of lipids used as candidates.)
    """
    
    parsed_annotations = ann.copy()

    ranked_lipids = {}
    is_ranked = {}
    utilized_sum_species = {}
    candidates = {}

    # Loop over ions in network
    for ion in ion_network.nodes():
        
        # In this case the ion did not have any connections in the network
        # and the method was not able to rank the lipids
        if all(ion_network.nodes[ion]['sum_species'].values == 0):
            
            is_ranked[ion] = False
            ranked_lipids[ion] = np.nan
            utilized_sum_species[ion] = "|".join(ion_network.nodes[ion]['sum_species'].index)
            candidates[ion] = np.nan
        else:
            ion_ranking = ion_network.nodes[ion]['sum_species']
            
            # Only show lipids in ranking which have had at least one neighbor
            ion_ranking = ion_ranking[ion_ranking > 0]
            
            is_ranked[ion] = True
            ranked_lipids[ion] = "|".join(ion_ranking.index)
            utilized_sum_species[ion] = "|".join(ion_network.nodes[ion]['sum_species'].index)
            candidates[ion] = np.array(ion_network.nodes[ion]['sum_species'].index)[:cutoff]

    parsed_annotations['is_ranked'] = pd.Series(is_ranked)
    parsed_annotations['utilized_sum_species'] = pd.Series(utilized_sum_species)                    
    parsed_annotations['ranked_lipids'] = pd.Series(ranked_lipids)
    parsed_annotations['candidates'] = pd.Series(candidates)

    return parsed_annotations
