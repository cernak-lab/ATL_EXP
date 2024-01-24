import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random
from collections import Counter
from data.data_utils import ADDITIVE_NAMES, ADDITIVE_IDS

DESC_NAMES = joblib.load("joblib_files/desc_names.joblib")


def train_target_models(target_X, target_y, n_models, n_trees, max_depth=1, sample_weight=None) : 
    """ Trains a random forest model based on target data. 
    Ensures that there are no decision trees of 'depth 0'. 
    
    Parameters
    ----------
    n_models : int
        Number of random forest models to train.

    n_trees : int
        Number of trees in each random forest to use in each model.
        
    max_depth : int
        Maximum depth of each decision tree.

    sample_weight : np.1darray of shape (n_rxns, )
        Weight of each sample. If None, uniform weight.
        
    Returns
    -------
    target_model_list : list of RandomForestClassifiers
        List of random forests.
    """
    target_model_list = []

    for i in range(n_models) :
        all_deep = False
        counter = 0
        while not all_deep :
            trees = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                random_state=42+i+100*counter,
                n_jobs=-1
            )
            trees.fit(target_X, target_y, sample_weight=sample_weight)
            depths = np.array([dtc.get_depth() for dtc in trees.estimators_])
            if np.prod(depths) != 0 :
                all_deep = True
            else : 
                counter += 1
        target_model_list.append(trees)
    
    return target_model_list


def get_descriptors_used_by_target_models(target_model_list, descriptor_name_list=DESC_NAMES) :
    """ Counts the descriptors used by decision trees.
    
    Parameters
    ----------
    target_model_list : list of RandomForestClassifiers
        List of target models

    descriptor_name_list : list of str
        List of descriptor names

    Returns
    -------
    desc_utilized : np.1darray of shape (n_descriptors, )
        Count of each descriptor being used by all decision trees in all RandomForestClassifiers in the list.
    """
    desc_utilized = np.zeros((len(descriptor_name_list)))

    for rfc in target_model_list :
        for dtc in rfc.estimators_ :
            desc_utilized[dtc.tree_.feature[0]] += 1
    
    return desc_utilized


def plot_freq_of_each_desc(desc_utilized, desc_names=DESC_NAMES):
    fig, ax = plt.subplots()
    plt.bar(np.arange(len(desc_utilized)), desc_utilized)
    for div_ind in [desc_names.index('Katritzky E_red'), desc_names.index('NiCl2'), desc_names.index('Ligand Ni-X'), desc_names.index('Cl'), desc_names.index('Solvent HansenD')] :
        ax.axvline(div_ind-0.5, 0, 1, ls="--", c='grey')
    ax.set_yticks(50 * np.arange(int(np.max(desc_utilized)//50 + 1)))
    ax.set_yticklabels(50 * np.arange(int(np.max(desc_utilized)//50 + 1)))
    ax.set_ylabel("Count", fontsize=14)
    ax.set_xlabel("Descriptor Index", fontsize=14)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)


def combine_models(source_model, target_model) :
    """ Combines two random forest models."""
    combined_model = copy.deepcopy(source_model)
    combined_model.estimators_ += target_model.estimators_
    combined_model.n_estimators += target_model.n_estimators
    return combined_model


def combine_two_model_list_shuffled(list_of_source_models, list_of_target_models, seed=42):
    """ First shuffles the list of target models. 
    Then, combines the source and target models at the same index.
    Assumes the same length between the lists."""
    random.seed(seed)
    combined_model_list = []
    copied_target_model_list = copy.deepcopy(list_of_target_models)
    random.shuffle(copied_target_model_list)
    combined_model_list = [
        combine_models(source_model, target_model)\
            for source_model, target_model in zip(list_of_source_models, copied_target_model_list)
    ]

    return combined_model_list


def print_suggestions(candidate_id_array, suggestion_counter, num_entries):
    """ Prints the top commonly suggested reactions from different models.
    
    Parameters
    ----------
    candidate_id_array : np.2darray of shape (n_rxns, n_ids)
        ID-array of reaction candidates.
    
    suggestion_counter : Counter
        Counter object that collected the number of which reactions are suggested by models.
    
    num_entries : int
        Number of suggestions to display.
    
    Returns
    -------
    None
    """
    sheets = ["Ni source", "Ligands", "Solvents"]
    dicts = []
    for sheet in sheets :
        df = pd.read_excel("data/descriptors.xlsx", sheet_name=sheet, usecols=[0,1])
        comp_dict = {}
        for i, row in df.iterrows():
            comp_dict.update({row[0]:row[1]})
        dicts.append(comp_dict)
    for rxn_id, freq in list(suggestion_counter.most_common(num_entries)) :
        print(f"[{dicts[0][candidate_id_array[rxn_id, 0]]:>12}, {dicts[1][candidate_id_array[rxn_id, 1]]:>5},\
            {ADDITIVE_NAMES[int(np.where(np.all(ADDITIVE_IDS == candidate_id_array[rxn_id, 2:4], axis=1))[0])]:>11},\
                  {dicts[2][candidate_id_array[rxn_id, 4]]:>7}] {freq}")


def count_num_topN_suggestions(list_of_list_of_models, candidate_desc_array, topN):
    """ Counts the frequency of each reaction making within the top-N highest probability to give positive yield.
    
    Parameters
    ----------
    list_of_list_of_models : list of list of Classifiers
        Different list groups to consider.
    
    candidate_desc_array : np.2darray of shape (n_rxns, n_descs)
        DESC-array of reaction candidates.

    topN : int
        The number of reactions to count for each model.

    Returns
    -------
    count_list : list of Counters
        Number of each reactions that become suggested within top N positive rxns for each set of models.
    """
    count_list = []
    for model_list in list_of_list_of_models:
        rxn_counter = Counter()
        for rfc in model_list :
            proba = rfc.predict_proba(candidate_desc_array)[:,1]
            best_rxns = np.argsort(proba)[::-1][:topN]
            rxn_counter.update(best_rxns)
        count_list.append(rxn_counter)
    return count_list


def plot_suggestion_distribution(n_trees_list, count_list):
    """ Plots the distribution of each reaction being suggested within the top-N highest probability of giving positive yield
    for each set of model.
    
    Parameters
    ----------
    n_trees_list : list of ints
        Number of decision trees used for making the Counter object.
        
    count_list : list of Counters
        Output of function above.
        
    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    for i, rxn_counter in enumerate(count_list) :
        count = sorted(list(rxn_counter.values()))
        print(count[-24:])
        bins = np.arange(len(count))
        ax.bar(bins, count, alpha=0.5, width=1, label=f"{n_trees_list[i]:3} Source Trees")
    ax.legend(loc="upper left")
    ax.set_yticks(np.arange(6)*20)
    ax.set_yticklabels(np.arange(6)*20)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_xlabel("Reaction Index", fontsize=14)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)


def plot_top_probability_distribution(list_of_models, candidate_desc_array, topN) :
    """ Makes boxplot of the probability values output by all models for
     each top-suggested reactions.
     
    Parameters
    ----------
    list_of_models : list of Classifiers
        Models to make predictions from.
    
    candidate_desc_array : np.2darray of shape (n_rxns, n_descs)
        DESC-array of reaction candidates.

    topN : int
        The number of reactions to count for each model.

    Returns
    -------
    proba_array : np.2darray (of shape (topN, n_models))
        All probability values output by every model for all reactions.
    """
    count_list = count_num_topN_suggestions([list_of_models], candidate_desc_array, topN)
    proba_array = np.zeros((topN, len(list_of_models)))
    vote_rank = []
    probability = []
    for i, (rxn_id, _) in enumerate(list(count_list[0].most_common(topN))) :
        vote_rank += [i+1] * len(list_of_models)
        for j, rfc in enumerate(list_of_models) : 
            proba_array[i, j] = rfc.predict_proba(candidate_desc_array[rxn_id, :].reshape(1,-1))[0,1]
            probability.append(proba_array[i,j])
    
    fig, ax = plt.subplots()
    sns.boxplot(
        data={"Vote Rank":vote_rank, "Probability":probability},
        x="Vote Rank",
        y="Probability",
        palette="viridis",
        ax=ax
    )
    ax.set_ylim(0,1)
    ax.set_yticks(0.2*np.arange(6))
    ax.set_yticklabels([round(0.2*x,1) for x in range(6)])
    ax.set_ylabel("Probability", fontsize=14)
    ax.set_xlabel("Rank by Votes", fontsize=14)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    plt.show()
    return proba_array


def plot_full_proba_dist(list_of_models, candidate_desc_array, log_scale=True) :
    """ Draws a lineplot of the average probability values predicted by all models for every reaction candidate.
    X axis is log-scaled due to the number of reactions.
    
    Parameters
    ----------
    list_of_models : list of Classifiers
        Models to make predictions from.
    
    candidate_desc_array : np.2darray of shape (n_rxns, n_descs)
        DESC-array of reaction candidates.

    Returns
    -------
    None
    """
    proba_array = np.zeros(candidate_desc_array.shape[0])
    for rfc in list_of_models : 
        proba_array += rfc.predict_proba(candidate_desc_array)[:,1]
    proba_array /= 100
    fig, ax = plt.subplots()
    ax.plot(sorted(proba_array)[::-1])
    if log_scale :
        ax.set_xscale("log")
    ax.set_ylim(0,1)
    ax.set_yticks([0.2*x for x in range(6)])
    ax.set_yticklabels([round(0.2*x,1) for x in range(6)])
    ax.set_ylabel("Average Predicted Probability", fontsize=14)
    ax.set_xlabel("Reaction Index", fontsize=14)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)