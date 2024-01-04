# %%
import random
from typing import Dict
import re
import pickle
import random

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_kernels
import scipy

import scanpy as sc
import linex2 as lx2
import linex2metaspace as lx2m
import os


# %% Helper functions
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
    return {n:(x + x_shift, 
               y + (y_shift * np.random.choice(np.array([-1,1])))) for n, (x,y) in pos.items()}

def match_lipid(x, pattern=r'^PC'):
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

def lipid_bubble_plot(likelilipids: pd.Series,
                      condition1: str,
                      condition2: str,
                      reference_lipids,
                      condition_data_dict: Dict[str, np.ndarray],
                      regex_pattern: str = r'\([0-9]+'
                     ):
    position_dict = {}

    for i in tqdm(range(len(likelilipids))):
        if likelilipids[i] is not np.nan:
            if match_lipid(likelilipids[i][0], pattern=regex_pattern):
                tmp = lx2.lipid_parser(likelilipids[i][0], reference_lipids=reference_lipids)
                tk = (tmp.sum_length(), tmp.sum_dbs())
                if tk in position_dict.keys():
                    position_dict[tk].append(i)
                else:
                    position_dict[tk] = [i]
            elif match_lipid(likelilipids[i][1], pattern=regex_pattern):
                tmp = lx2.lipid_parser(likelilipids[i][1], reference_lipids=reference_lipids)
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

    df1 = pd.DataFrame(mean_dict_c1.items()).rename(columns={0:'Pos', 1: 'C1'})
    df1['C'] = df1['Pos'].apply(lambda x: x[0])
    df1['DB'] = df1['Pos'].apply(lambda x: x[1])
    df2 = pd.DataFrame(mean_dict_c2.items()).rename(columns={0:'Pos', 1: 'C2'})
    df2['C'] = df2['Pos'].apply(lambda x: x[0])
    df2['DB'] = df2['Pos'].apply(lambda x: x[1])
    final_df = pd.concat([df1.set_index(['C', 'DB']).drop(columns=['Pos']), 
                          df2.set_index(['C', 'DB']).drop(columns=['Pos'])], axis=1).reset_index()
    final_df['LogFC'] = final_df['C1'] - final_df['C2']
    final_df['|LogFC|'] = abs(final_df['C1'] - final_df['C2'])
    
    return final_df

# Test if even chain lenghts are scoring higher on average
def iseven(num):
    return (num % 2) == 0

def scaling_position(x, positions):
    if positions==1:
        return 0
    else:
        return x/(positions-1)

def get_ranking_dict(net):
    out_dict = {}

    # K - ion, V - data
    for k, v in dict(net.nodes(data=True)).items():
        sum_species = v['sum_species']
        # Get exemplary lipids for each sum species
        tmp_dict = {}
        for pos, ss, score in zip(range(len(sum_species)), sum_species.index, sum_species.values):
            # find a matching lipid
            for ll in v['parsed_lipids']:
                if ll.sum_species_str() == ss:
                    
                    # Include only actually ranked lipids
                    if not (sum_species==0).all():
                        tmp_dict[ss] = {
                            'relpos': scaling_position(pos, len(sum_species)), # FIX SCALING 
                            'abspos': pos,
                            'score': score,
                            'lipid': ll,
                            'iseven': iseven(ll.sum_length()),
                            'hasether': ll.get_ether()
                                        }
                    break
        out_dict[k] = tmp_dict
    
    return out_dict


# %% cell3
class ComposeAnnData:
    pass

sc_tab = pd.read_csv('data/hela_nih3t3/spaceM_source_data.csv') \
    .rename(columns={'Cell type label (HeLa, 0; NIH3T3, 1)': 'celltype'})

sc_tab['celltype'] = sc_tab['celltype'].apply(lambda x: 'Hela' if x==0 else 'NIH3T3')

# %%
sns.scatterplot(sc_tab, x='UMAP X', y='UMAP Y', hue='celltype')



# %%
sc_tab.columns = [x.split(' - ')[0] for x in sc_tab.columns]
obs = sc_tab[['Cell Index', 'UMAP X', 'UMAP Y', 'celltype']]
annotations = pickle.load(open('data/hela_nih3t3/sm_results_2018-05-29_11h23m47s.pickle', 'rb'))


tmp = list(annotations.reset_index().set_index('formula').index)
mapped_metabolites = []
for x in list(sc_tab.drop(columns=['Cell Index', 'UMAP X', 'UMAP Y', 'celltype']).columns):
    if x in tmp:
        mapped_metabolites.append(x)
matrx = np.array(sc_tab.drop(columns=['Cell Index', 'UMAP X', 'UMAP Y', 'celltype']))
f"{str(len(mapped_metabolites))} of {matrx.shape[1]} metabolites were successfully mapped"

adata = sc.AnnData(
    X=np.array(sc_tab.drop(columns=['Cell Index', 'UMAP X', 
                                    'UMAP Y', 'celltype']).loc[:,mapped_metabolites], dtype='float32'),
    obs=obs,
    var=annotations.loc[mapped_metabolites, :].reset_index()
)
adata.var = adata.var.set_index(['formula', 'adduct'])


# %%
feature_sim = pd.DataFrame(pairwise_kernels(np.array(adata.X).transpose(), metric='cosine'))
feature_sim.columns = adata.var.index
feature_sim.index = adata.var.index
# %%
feature_sim[('C18H32O2', '-H')][('C19H37O7P', '-H')]
# %%

# %%
class ComputeNetwork:
    pass

ref_lip_dict = lx2m.get_lx2_ref_lip_dict()
class_reacs = lx2m.get_organism_combined_class_reactions(ref_lip_dict=ref_lip_dict, organism='HSA')
# %%
annotations = adata.var
parsed_lipids = lx2m.parse_annotation_series(annotations['moleculeNames'], ref_lip_dict)
keep_annotations = lx2m.annotations_parsed_lipids(parsed_lipids)
parsed_annotations = annotations.copy()
parsed_annotations['parsed_lipids'] = parsed_lipids
parsed_annotations = parsed_annotations.loc[keep_annotations,:]


# %%
bootstraps = 80

g = lx2m.bootstrap_networks(
        lx2m.unique_sum_species(parsed_annotations['parsed_lipids']),
        parsed_annotations['parsed_lipids'],
        n=bootstraps,
        lx2_class_reacs=class_reacs,
        lx2_reference_lipids=lx2m.get_lx2_ref_lips(),
        return_composed=True
    )

# %%
ig = lx2m.ion_weight_graph(g,  # type: ignore
                           lx2m.unique_sum_species(parsed_annotations['parsed_lipids']), 
                           bootstraps=bootstraps,
                           parsed_lipids=parsed_annotations['parsed_lipids'],
                           #feature_similarity=feature_sim
                          )


# %%
class RunMultipleNetworks:
    pass

ig_evaluation_dict = {}
# Evaluate a range of number of bootstraps
# Start with 10, if the #bootstrap evaluation is required
for bt in tqdm(range(10, 51, 10)):
    
    ig_evaluation_dict[bt] = []
    # 10 Repetitions per number of bootstraps
    for i in range(10):
        tmp_g = lx2m.bootstrap_networks(
                    lx2m.unique_sum_species(parsed_annotations['parsed_lipids']),
                    parsed_annotations['parsed_lipids'],
                    n=bt,
                    lx2_class_reacs=class_reacs,
                    lx2_reference_lipids=lx2m.get_lx2_ref_lips(),
                    return_composed=True,
                    print_iterations=False
                )

        tmp_ig = lx2m.ion_weight_graph(
                    tmp_g,  # type: ignore
                    lx2m.unique_sum_species(parsed_annotations['parsed_lipids']), 
                    bootstraps=bt,
                    parsed_lipids=parsed_annotations['parsed_lipids'],
                    #feature_similarity=feature_sim
                 )
        
        ig_evaluation_dict[bt].append(tmp_ig)




# %%
class ComparisonToLCMSdata:
    pass

bulk_pos = pd.read_csv('data/hela_nih3t3/LCMS_pos.csv')
bulk_neg = pd.read_csv('data/hela_nih3t3/LCMS_neg.csv')

bulk_pos = bulk_pos[['Metabolite name', 'Blank_prep_no_IS_Pos_0uL'] + 
                    [x for x in bulk_pos.columns if x.startswith('Luca140818')]]
bulk_neg = bulk_neg[['Metabolite name', 'Blank_prep_no_IS_Neg_0uL'] + 
                    [x for x in bulk_neg.columns if x.startswith('Luca140818')]]

bulk_pos[['sum_species', 'molecular_species']] = bulk_pos['Metabolite name'].str.split('|', expand=True)
bulk_neg[['sum_species', 'molecular_species']] = bulk_neg['Metabolite name'].str.split('|', expand=True)

def uln(lip):
    tmp = lip.split(' ')
    if len(tmp)==1:
        return lip
    else:
        return f'{tmp[0]}({tmp[1]})'

# Update lipid nomenclature
bulk_pos['sum_species'] = bulk_pos['sum_species'].apply(uln)
bulk_neg['sum_species'] = bulk_neg['sum_species'].apply(uln)

identified_sum_species = set(list(bulk_neg['sum_species']) + list(bulk_pos['sum_species']))

out_dict = {}

# K - ion, V - data
for k, v in dict(ig.nodes(data=True)).items():
    sum_species = v['sum_species']
    # Get exemplary lipids for each sum species
    tmp_dict = {}
    for pos, ss, score in zip(range(len(sum_species)), sum_species.index, sum_species.values):
        # find a matching lipid
        for ll in v['parsed_lipids']:
            if ll.sum_species_str() == ss:
                break
        # Include only actually ranked lipids
        if not (sum_species==0).all():
            tmp_dict[ss] = {
                'relpos': scaling_position(pos, len(sum_species)), # FIX SCALING OF POSITIONS
                'abspos': pos,
                'score': score,
                'lipid': ll, # type: ignore
                'iseven': iseven(ll.sum_length()), # type: ignore
                'hasether': ll.get_ether() # type: ignore
                           }
    out_dict[k] = tmp_dict

# Evaluate
# Only for cases where even and odd chain lipid in annotation

conf_pos = []
nconf_pos = []

# Loop over all ions
for k, v in out_dict.items():
    
    # Loop over all ranked sum_species per ion
    for ss in v.keys():
        if ss in identified_sum_species:
            conf_pos.append(v[ss]['relpos'])
        else:
            nconf_pos.append(v[ss]['relpos'])
        
conf_df = pd.concat([pd.DataFrame({'pos': conf_pos, 'type': 'Confirmed lipids'}), 
                     pd.DataFrame({'pos': nconf_pos, 'type': 'Not confirmed lipids'})]).reset_index()


# %% 
class HelperFunctions:
    pass

def calculate_confusion_matrix(ranking_dict, cut_off_function, ground_truth, **kwargs):
    out = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    
    for ion, ranking in ranking_dict.items():
        tmp = cut_off_function(ranking, **kwargs)
        
        for ss, indication in tmp.items():
            if ss in ground_truth and indication==1:
                out['tp'] +=1
            elif ss not in ground_truth and indication==1:
                out['fp'] +=1
            elif ss not in ground_truth and indication==0:
                out['tn'] +=1
            elif ss in ground_truth and indication==0:
                out['fn'] +=1
    
    return out

def first_cutoff(ranking):
    return {ss: 1 if info['relpos']==0 else 0 for ss, info in ranking.items()}

def abs_cutoff(ranking, position=1):
    return {ss: 1 if info['abspos']<=position else 0 for ss, info in ranking.items()}

def scoring_cutoff(ranking, fraction=0.3):
    total_score = sum([info['score'] for info in ranking.values()])
    
    return {ss: 1 if info['score']/total_score>=fraction else 0 for ss, info in ranking.items()}

def randomly_picking(ranking, num=1):
    tmp = list(ranking.keys())
    if (len(tmp)>0):
        if len(tmp)>=num:
            choice = random.sample(tmp, k=num)
        else:
            choice = random.sample(tmp, k=len(tmp))
    else:
        choice = ''
    return {ss: 1 if ss in choice else 0 for ss, info in ranking.items()}
        
def sensitivity(x):
    return x['tp'] / (x['tp'] + x['fn'])

def specificity(x):
    return x['tn'] / (x['tn'] + x['fp'])

def accuracy(x):
    return (x['tp'] + x['tn'])/(x['fn']+x['tn']+x['fp']+x['tp'])

def f1score(x):
    return (x['tp']*2)/(x['tp']*2 + x['fp'] + x['fn'])

def all_scores(x, metric, verbose = True):
    return_dict = {
        'sensitivity': sensitivity(x),
        'specificity': specificity(x),
        'accuracy': accuracy(x),
        'f1score': f1score(x)
                  }
    if verbose:
        print(f'{metric}:')
        print(f'- sensitivity: {round(return_dict["sensitivity"], 3)}')
        print(f'- specificity: {round(return_dict["specificity"], 3)}')
        print(f'- accuracy: {round(return_dict["accuracy"], 3)}')
        print(f'- F1-score: {round(return_dict["f1score"], 3)}')
        
    return return_dict

def eval_scores(ig_evaluation_dict, absolute_position=1, scoring_fraction=0.7, random_num=1):
    eval_dict = {'bootstraps': [], 'sensitivity': [], 'specificity': [], 
                 'accuracy': [], 'f1score': [], 'cut-off metric': []}

    # Loop over bootstraps
    for bt, nets in ig_evaluation_dict.items():
        # Loop over repetitions
        for net in nets:
            ranking_dict = get_ranking_dict(net)

            # First cut-off
            confmat = calculate_confusion_matrix(ranking_dict, first_cutoff, identified_sum_species)
            scores = all_scores(confmat, '', verbose=False)
            eval_dict['bootstraps'].append(bt)
            eval_dict['sensitivity'].append(scores['sensitivity'])
            eval_dict['specificity'].append(scores['specificity'])
            eval_dict['accuracy'].append(scores['accuracy'])
            eval_dict['f1score'].append(scores['f1score'])
            eval_dict['cut-off metric'].append('First cut-off')

            # Absolute cut-off
            confmat = calculate_confusion_matrix(ranking_dict, abs_cutoff, identified_sum_species, 
                                                 position=absolute_position)
            scores = all_scores(confmat, '', verbose=False)
            eval_dict['bootstraps'].append(bt)
            eval_dict['sensitivity'].append(scores['sensitivity'])
            eval_dict['specificity'].append(scores['specificity'])
            eval_dict['accuracy'].append(scores['accuracy'])
            eval_dict['f1score'].append(scores['f1score'])
            eval_dict['cut-off metric'].append('Absolute cut-off')

            # Scoring cut-off
            confmat = calculate_confusion_matrix(ranking_dict, scoring_cutoff, identified_sum_species, 
                                                 fraction=scoring_fraction)
            scores = all_scores(confmat, '', verbose=False)
            eval_dict['bootstraps'].append(bt)
            eval_dict['sensitivity'].append(scores['sensitivity'])
            eval_dict['specificity'].append(scores['specificity'])
            eval_dict['accuracy'].append(scores['accuracy'])
            eval_dict['f1score'].append(scores['f1score'])
            eval_dict['cut-off metric'].append('Scoring cut-off')

            # Random cut-off
            random_dict = {'sensitivity': [], 'specificity': [], 'accuracy': [],'f1score': []}
            for i in range(20):
                confmat = calculate_confusion_matrix(ranking_dict, randomly_picking, identified_sum_species, 
                                                     num=random_num)
                tmp = all_scores(confmat, 'Random', verbose=False)
                random_dict['sensitivity'].append(tmp['sensitivity'])
                random_dict['specificity'].append(tmp['specificity'])
                random_dict['accuracy'].append(tmp['accuracy'])
                random_dict['f1score'].append(tmp['f1score'])

            final_dict = {k: np.mean(v) for k, v in random_dict.items()}
            eval_dict['bootstraps'].append(bt)
            eval_dict['sensitivity'].append(final_dict['sensitivity'])
            eval_dict['specificity'].append(final_dict['specificity'])
            eval_dict['accuracy'].append(final_dict['accuracy'])
            eval_dict['f1score'].append(final_dict['f1score'])
            eval_dict['cut-off metric'].append('Random cut-off')

    eval_df = pd.melt(pd.DataFrame(eval_dict), 
                      id_vars=['bootstraps', 'cut-off metric'], 
                      value_vars=['sensitivity', 'specificity', 'accuracy', 'f1score'],
                      var_name='score')
    return eval_df



# %%
def ax_ann(fig, ax, letter='A', xoffset=0., yoffset=0., **kwargs):

    axes_position = ax.get_position()
    # Extract position and size information
    left, bottom, width, height = axes_position.bounds
    
    fig.text(left+xoffset, bottom+height+yoffset, letter, **kwargs)


XXSMALL_SIZE = 5
XSMALL_SIZE = 6
SMALLSMALL_SIZE = 8
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 18
cm = 1/2.54

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLSMALL_SIZE, title_fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title





# %%
# ##################
# MAIN FIGURE
# ##################




fig = plt.figure(figsize=(30*cm, 30*cm))

gs = plt.GridSpec(nrows=5, ncols=6) # type: ignore

sketchax = fig.add_subplot(gs[0:3, 0:3])
netax = fig.add_subplot(gs[0:3, 3:6])

posax = fig.add_subplot(gs[3:5, 0:2])
confax = fig.add_subplot(gs[3:5, 2:4])
eval1ax = fig.add_subplot(gs[3, 4])
eval2ax = fig.add_subplot(gs[3, 5])
eval3ax = fig.add_subplot(gs[4, 4])
eval4ax = fig.add_subplot(gs[4, 5])

fig.subplots_adjust(left=0, right=1.1, hspace=.3, wspace=.6)

ax = sketchax
img = image.imread('figures/schematic.png')
ax.imshow(img)
ax.axis('off')
ax_ann(fig, ax, letter='A', size=BIGGER_SIZE, weight='bold', xoffset=-.02)


ax = netax
pos = nx.spring_layout(ig, k=.35)
nx.draw_networkx_nodes(ig, pos=pos, node_size=10, ax=ax)
nx.draw_networkx_edges(ig, pos=pos, 
                       width=[d['weight']/2 for u, v, d in ig.edges(data=True)], # type: ignore
                       ax=ax)
nx.draw_networkx_labels(ig, pos=pos,
                        labels={k: ", ".join(v['sum_species'].index) for k, v in dict(ig.nodes(data=True)).items()},
                        font_size=6, ax=ax)
ax.axis('off')
ax_ann(fig, ax, letter='B', size=BIGGER_SIZE, weight='bold')


ax = posax
df_l = []
counter=0
for k, v in dict(ig.nodes(data=True)).items():
    if len(v['sum_species'])>1:
        df_l.append(pd.DataFrame({'position': list(range(len(v['sum_species']))), 
                                  'proportion':v['sum_species'].values/sum(v['sum_species']),
                                  'node': str(counter)
                                 }))
    counter+=1

sns.lineplot(pd.concat(df_l).reset_index(), x='position', y='proportion', hue='node', ax=ax, legend=False)
sns.violinplot(pd.concat(df_l).reset_index(), x='position', y='proportion', ax=ax, legend=False, color='white')
ax.set_xlabel('Ranking position')
ax.set_ylabel('Proportion of edges per annotation')
sns.despine(offset=5, trim=False, ax=ax)
ax_ann(fig, ax, letter='C', size=BIGGER_SIZE, weight='bold', xoffset=-.08)


ax = confax
sns.violinplot(conf_df, x='type', y='pos', scale='count', palette={'Confirmed lipids': '#18974C', 
                                                                   'Not confirmed lipids': '#734595'}, 
                                                                   ax=ax)
ax.annotate('', xy=(0, 1.5), xytext=(1,1.5), arrowprops=dict(arrowstyle='-', color='black'))

testres = scipy.stats.mannwhitneyu(conf_pos, nconf_pos).pvalue
sigind = 'n.s.'
sigind = '*' if testres <= 0.05 else sigind
sigind = '**' if testres <= 0.01 else sigind
sigind = '***' if testres <= 0.001 else sigind
ax.annotate(sigind, xy=(0.5, 1.51), ha='center', fontsize=XSMALL_SIZE)
ax.set_ylabel('Relative lipid ranking')
ax.set_xlabel('')
ax.set_ylim((-.5, 1.6))
sns.despine(offset=5, trim=False, ax=ax)
ax_ann(fig, ax, letter='D', size=BIGGER_SIZE, weight='bold', xoffset=-.08)


eval_df = eval_scores(ig_evaluation_dict, random_num=1, scoring_fraction=0.5, absolute_position=0)

eval_df['cut-off metric'] = eval_df['cut-off metric'].replace(
    ['First cut-off', 'Absolute cut-off', 'Scoring cut-off', 'Random cut-off'], 
    ['Most likely', 'Most likely', '50% of neighbors', 'Random 2 molecules'])
eval_50 = eval_df[eval_df['bootstraps']==50]

# Sensitivity
ax = eval1ax
tmp = eval_50[eval_50['score']=='sensitivity']
sns.boxplot(data=tmp, x='cut-off metric', y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xticklabels([], rotation=30, ha='right')
ax.set_title('Sensitivity')
ax.set_xlabel('')
ax.set_ylabel('')
ax_ann(fig, ax, letter='E', size=BIGGER_SIZE, weight='bold', xoffset=-.08)

# Specificity
ax = eval2ax
tmp = eval_50[eval_50['score']=='specificity']
sns.boxplot(data=tmp, x='cut-off metric', y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xticklabels([], rotation=30, ha='right')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Specificity')

# Accuracy
ax = eval3ax
tmp = eval_50[eval_50['score']=='accuracy']
sns.boxplot(data=tmp, x='cut-off metric', y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax.set_title('Accuracy')
ax.set_ylabel('')
ax.set_xlabel('Cut-off metric')

# F1-score
ax = eval4ax
tmp = eval_50[eval_50['score']=='f1score']
sns.boxplot(data=tmp, x='cut-off metric', y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax.set_title('F1-score')
ax.set_ylabel('')
ax.set_xlabel('Cut-off metric')

plt.show()



# %%
# ##################
# Supplementary Figure 1
# * Self evaluation
# ##################
out_dict = get_ranking_dict(ig)

avg_even_pos = []
avg_odd_pos = []
for k, v in out_dict.items():
    if not all([v2['iseven'] for v2 in v.values()]) and not all([not v2['iseven'] for v2 in v.values()]):
        # Get average even pos
        avg_even_pos.append(np.mean([v2['relpos'] for v2 in v.values() if v2['iseven']]))
        # get average odd pos
        avg_odd_pos.append(np.mean([v2['relpos'] for v2 in v.values() if not v2['iseven']]))
        
eo_df = pd.concat([pd.DataFrame({'pos': avg_even_pos, 'type': 'even'}), 
                   pd.DataFrame({'pos': avg_odd_pos, 'type': 'odd'})]).reset_index()
        
avg_ether_pos = []
avg_nonether_pos = []
for k, v in out_dict.items():
    if not all([v2['hasether']==0 for v2 in v.values()]) and not all([v2['hasether']>0 for v2 in v.values()]):
        # Get ether
        avg_ether_pos.append(np.mean([v2['relpos'] for v2 in v.values() if v2['hasether']>0]))
        # Get non ether
        avg_nonether_pos.append(np.mean([v2['relpos'] for v2 in v.values() if v2['hasether']==0]))
        
ether_df = pd.concat([pd.DataFrame({'pos': avg_ether_pos, 'type': 'ether'}), 
                   pd.DataFrame({'pos': avg_nonether_pos, 'type': 'ester'})]).reset_index()


# %%
fig = plt.figure(figsize=(30*cm, 20*cm))
fig.subplots_adjust(wspace=.6, hspace=.3)

gs = plt.GridSpec(nrows=2, ncols=3) # type: ignore
netax = fig.add_subplot(gs[0:2, 0:2])
esterax = fig.add_subplot(gs[0, 2])
evenax = fig.add_subplot(gs[1, 2])




ax = netax
pos = nx.spring_layout(g, k=1.5) # type: ignore
nx.draw_networkx_nodes(g, pos=pos, node_size=10, ax=ax) # type: ignore
nx.draw_networkx_edges(g, pos=pos, width=.01, ax=ax) # type: ignore
nx.draw_networkx_labels(g, pos=pos, font_size=XXSMALL_SIZE, ax=ax) # type: ignore
ax.axis('off')
ax_ann(fig, ax, letter='A', size=BIGGER_SIZE, weight='bold', xoffset=-.00)



ax = esterax
sns.violinplot(eo_df, x='type', y='pos', palette={'even': '#18974C', 'odd': '#734595'}, 
               scale='area', ax=ax)
ax.set_ylabel('Relative ranking')
ax.set_xlabel('Lipid chain length')
sns.despine(offset=5, trim=False, ax=ax)
ax_ann(fig, ax, letter='B', size=BIGGER_SIZE, weight='bold', xoffset=-.08)


ax = evenax
sns.violinplot(ether_df, x='type', y='pos', palette={'ester': '#18974C', 'ether': '#734595'}, 
               scale='area', ax=ax)
ax.set_ylabel('Relative ranking')
ax.set_xlabel('Lipid type')
sns.despine(offset=5, trim=False, ax=ax)
ax_ann(fig, ax, letter='C', size=BIGGER_SIZE, weight='bold', xoffset=-.08)

plt.show()


# %%
# ##################
# Supplementary Figure 2
# * Cut-off metric selection
# ##################

# precomputing data

# Abosolute cut-off
eval_dict = {'maxpos': [], 'sensitivity': [], 'specificity': [], 
                 'accuracy': [], 'f1score': [], 'cut-off metric': []}

# Loop over repetitions
for net in ig_evaluation_dict[50]:
    ranking_dict = get_ranking_dict(net)
    for i in range(5):
        # Absolute cut-off
        confmat = calculate_confusion_matrix(ranking_dict, abs_cutoff, identified_sum_species, 
                                             position=i)
        scores = all_scores(confmat, '', verbose=False)
        eval_dict['maxpos'].append(i+1)
        eval_dict['sensitivity'].append(scores['sensitivity'])
        eval_dict['specificity'].append(scores['specificity'])
        eval_dict['accuracy'].append(scores['accuracy'])
        eval_dict['f1score'].append(scores['f1score'])
        eval_dict['cut-off metric'].append('Absolute cut-off')
        
abs_df = pd.melt(pd.DataFrame(eval_dict), 
                 id_vars=['maxpos', 'cut-off metric'], 
                 value_vars=['sensitivity', 'specificity', 'accuracy', 'f1score'],
                 var_name='score')

# Neightbor cur-off
eval_dict = {'fraction': [], 'sensitivity': [], 'specificity': [], 
                 'accuracy': [], 'f1score': [], 'cut-off metric': []}

cos = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

# Loop over repetitions
for net in ig_evaluation_dict[50]:
    ranking_dict = get_ranking_dict(net)
    for i in cos:
        # Absolute cut-off
        confmat = calculate_confusion_matrix(ranking_dict, scoring_cutoff, identified_sum_species, 
                                                 fraction=i)
        scores = all_scores(confmat, '', verbose=False)
        eval_dict['fraction'].append(i)
        eval_dict['sensitivity'].append(scores['sensitivity'])
        eval_dict['specificity'].append(scores['specificity'])
        eval_dict['accuracy'].append(scores['accuracy'])
        eval_dict['f1score'].append(scores['f1score'])
        eval_dict['cut-off metric'].append('Scoring cut-off')
        
ngb_df = pd.melt(pd.DataFrame(eval_dict), 
                      id_vars=['fraction', 'cut-off metric'], 
                      value_vars=['sensitivity', 'specificity', 'accuracy', 'f1score'],
                      var_name='score')
ngb_df['fraction'] = ngb_df['fraction'].astype(str)


# Random cut-off
eval_dict = {'maxpos': [], 'sensitivity': [], 'specificity': [], 
                 'accuracy': [], 'f1score': [], 'cut-off metric': []}

# Loop over repetitions
for net in ig_evaluation_dict[50]:
    ranking_dict = get_ranking_dict(net)
    for i in range(5):
        # Absolute cut-off
        random_dict = {'sensitivity': [], 'specificity': [], 'accuracy': [],'f1score': []}
        for j in range(20):
            confmat = calculate_confusion_matrix(ranking_dict, randomly_picking, identified_sum_species, 
                                                 num=i)
            tmp = all_scores(confmat, 'Random', verbose=False)
            random_dict['sensitivity'].append(tmp['sensitivity'])
            random_dict['specificity'].append(tmp['specificity'])
            random_dict['accuracy'].append(tmp['accuracy'])
            random_dict['f1score'].append(tmp['f1score'])

        final_dict = {k: np.mean(v) for k, v in random_dict.items()}
        eval_dict['maxpos'].append(i+1)
        eval_dict['sensitivity'].append(final_dict['sensitivity'])
        eval_dict['specificity'].append(final_dict['specificity'])
        eval_dict['accuracy'].append(final_dict['accuracy'])
        eval_dict['f1score'].append(final_dict['f1score'])
        eval_dict['cut-off metric'].append('Random cut-off')
        
rand_df = pd.melt(pd.DataFrame(eval_dict), 
                      id_vars=['maxpos', 'cut-off metric'], 
                      value_vars=['sensitivity', 'specificity', 'accuracy', 'f1score'],
                      var_name='score')

# %%
fig = plt.figure(figsize=(30*cm, 20*cm))
fig.subplots_adjust(hspace=.6, left=0)

axs = fig.subplots(3, 4)

# ##
# Absolute
# ##
axx = axs[0, :] # type: ignore
df = abs_df
x='maxpos'
x_title = 'Maximum position'

# Sensitivity
ax = axx[0]
tmp = df[df['score']=='sensitivity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Sensitivity')
ax.set_xlabel(x_title)
ax.set_ylabel('')
ax_ann(fig, ax, letter='A', size=BIGGER_SIZE, weight='bold', xoffset=-.08)

# Specificity
ax = axx[1]
tmp = df[df['score']=='specificity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xlabel(x_title)
ax.set_ylabel('')
ax.set_title('Specificity')

# Accuracy
ax = axx[2]
tmp = df[df['score']=='accuracy']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Accuracy')
ax.set_ylabel('')
ax.set_xlabel(x_title)

# F1-score
ax = axx[3]
tmp = df[df['score']=='f1score']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('F1-score')
ax.set_ylabel('')
ax.set_xlabel(x_title)

# ##
# Neighbors
# ##
axx = axs[1, :] # type: ignore
df = ngb_df
x='fraction'
x_title = 'Neighbor fraction'

# Sensitivity
ax = axx[0]
tmp = df[df['score']=='sensitivity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Sensitivity')
ax.set_xlabel(x_title)
ax.set_ylabel('')
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax_ann(fig, ax, letter='B', size=BIGGER_SIZE, weight='bold', xoffset=-.08)

# Specificity
ax = axx[1]
tmp = df[df['score']=='specificity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xlabel(x_title)
ax.set_ylabel('')
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax.set_title('Specificity')

# Accuracy
ax = axx[2]
tmp = df[df['score']=='accuracy']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Accuracy')
ax.set_ylabel('')
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax.set_xlabel(x_title)

# F1-score
ax = axx[3]
tmp = df[df['score']=='f1score']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('F1-score')
ax.set_ylabel('')
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax.set_xlabel(x_title)


# ##
# Absolute
# ##
axx = axs[2, :] # type: ignore
df = rand_df
x='maxpos'
x_title = 'Maximum position'

# Sensitivity
ax = axx[0]
tmp = df[df['score']=='sensitivity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Sensitivity')
ax.set_xlabel(x_title)
ax.set_ylabel('')
ax_ann(fig, ax, letter='C', size=BIGGER_SIZE, weight='bold', xoffset=-.08)

# Specificity
ax = axx[1]
tmp = df[df['score']=='specificity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xlabel(x_title)
ax.set_ylabel('')
ax.set_title('Specificity')

# Accuracy
ax = axx[2]
tmp = df[df['score']=='accuracy']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Accuracy')
ax.set_ylabel('')
ax.set_xlabel(x_title)

# F1-score
ax = axx[3]
tmp = df[df['score']=='f1score']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('F1-score')
ax.set_ylabel('')
ax.set_xlabel(x_title)

plt.show()


# %%
# ##################
# Supplementary Figure 3
# * Influence bootstraps
# ##################

fig = plt.figure(figsize=(30*cm, 20*cm))
fig.subplots_adjust(hspace=.6, left=0)

axs = fig.subplots(3, 4)

# ##
# 10
# ##
axx = axs[0, :] # type: ignore
df = eval_df[eval_df['bootstraps']==10]
x='cut-off metric'
x_title = 'Cut-off metric'

# Sensitivity
ax = axx[0]
tmp = df[df['score']=='sensitivity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Sensitivity')
ax.set_xlabel(x_title)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels([], rotation=30, ha='right')
ax_ann(fig, ax, letter='A', size=BIGGER_SIZE, weight='bold', xoffset=-.08)

# Specificity
ax = axx[1]
tmp = df[df['score']=='specificity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xlabel(x_title)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels([], rotation=30, ha='right')
ax.set_title('Specificity')

# Accuracy
ax = axx[2]
tmp = df[df['score']=='accuracy']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Accuracy')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels([], rotation=30, ha='right')

# F1-score
ax = axx[3]
tmp = df[df['score']=='f1score']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('F1-score')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels([], rotation=30, ha='right')

# ##
# 30
# ##
axx = axs[1, :] # type: ignore
df = eval_df[eval_df['bootstraps']==30]
x='cut-off metric'
x_title = 'Cut-off metric'

# Sensitivity
ax = axx[0]
tmp = df[df['score']=='sensitivity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Sensitivity')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels([], rotation=30, ha='right')
ax_ann(fig, ax, letter='B', size=BIGGER_SIZE, weight='bold', xoffset=-.08)

# Specificity
ax = axx[1]
tmp = df[df['score']=='specificity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels([], rotation=30, ha='right')
ax.set_title('Specificity')

# Accuracy
ax = axx[2]
tmp = df[df['score']=='accuracy']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Accuracy')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels([], rotation=30, ha='right')

# F1-score
ax = axx[3]
tmp = df[df['score']=='f1score']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('F1-score')
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels([], rotation=30, ha='right')


# ##
# 50
# ##
axx = axs[2, :] # type: ignore
df = eval_df[eval_df['bootstraps']==50]
x='cut-off metric'
x_title = 'Cut-off metric'

# Sensitivity
ax = axx[0]
tmp = df[df['score']=='sensitivity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Sensitivity')
ax.set_xlabel(x_title)
ax.set_ylabel('')
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax_ann(fig, ax, letter='C', size=BIGGER_SIZE, weight='bold', xoffset=-.08)

# Specificity
ax = axx[1]
tmp = df[df['score']=='specificity']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_xlabel(x_title)
ax.set_ylabel('')
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax.set_title('Specificity')

# Accuracy
ax = axx[2]
tmp = df[df['score']=='accuracy']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('Accuracy')
ax.set_ylabel('')
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax.set_xlabel(x_title)

# F1-score
ax = axx[3]
tmp = df[df['score']=='f1score']
sns.boxplot(data=tmp, x=x, y='value', color='grey', ax=ax)
sns.despine(offset=5, trim=False, ax=ax)
xtl = ax.get_xticklabels()
ax.set_title('F1-score')
ax.set_ylabel('')
xtl = ax.get_xticklabels()
ax.set_xticklabels(xtl, rotation=30, ha='right')
ax.set_xlabel(x_title)




# %%
# ##################
# Supplementary Figure 4
# * Application on spatial Data from METASPACE
# ##################