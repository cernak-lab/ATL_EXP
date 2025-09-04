# Prospective Active Transfer Learning on the Formal Coupling of Amines and Carboxylic Acids to Form Seconday Alkyl bonds
Code to reproduce the paper.

## Core libraries and their versions used in this repo
* numpy = 1.23.1
* pandas = 1.4.4
* matplotlib = 3.5.2
* seaborn = 0.12.0
* scikit-learn = 1.1.1
* shap = 0.45.1

## Contents
1. data
* reaction_data.xlsx: file with all the reaction sorted by the product.
* descriptors.xlsx: file with all the physical descriptors.
* data_utils.py: includes code snippets that prepares/transforms arrays of reactions with different representations.
* prepare_array.py: code that produces initial arrays of source reaction data.
* eda.ipynb: exploratory data analysis, corresponds to sections 5-2 and 5–3 in the SI.

2. joblib_files
* X_desc.joblib, X_id.joblib, y.joblib: source dataset arrays.
* desc_names.joblib: contains the list of names of descriptors used for each reaction component.
* source_model_N_trees.joblib: initial set of source models with different number of trees. used as a starting point for modeling.

3. preliminary_modeling.ipynb
Explores various combinations of n_estimators and max_depth for random forests on their effect on reaction selection. Corresponds to section 5-5 in the SI.

4. atl_utils.py
Includes all the helper functions to carry out ATL and their analyses. 

5. Ala4MeBn.ipynb, ProMex.ipynb, ProIndan.ipynb, AlaIndan.ipynb
Includes all the reaction selection processes, model updates and their analyses.


## Workflow for adapting code-base for future use
Code in this repo was written specifcally for the transformation investigated in this work.
Below describes the jupyter notebooks in more detail so future users could modify the code in this repo to apply to their reaction.

First, please take a lookg at `reaction_data.xlsx` and `descriptors.xlsx` to get a sense of how the reaction data was kept track of. Data organization for a new reaction in this way would make it easier for this code-base to be applied.

1. Preparing descriptor array of the target reaction
* These are necessary as inputs for the machine learning model/
* Descriptor array of all candidate *reaction conditions* (each row=single unique reaction condition; each column=descriptor) are enumerated using `prep_array_of_enumerated_candidates` in `data_utils.py`. A list of descriptor arrays of each reaction component (e.g., catalyst, ligand or solvent) needs to be provided as input.
```python
X_candidate_desc = prep_array_of_enumerated_candidates([catalyst_descriptors, solvent_descriptors])
```
* Then the descriptors of the target substrate (pair) is stacked with the descriptor array of reaction condition.
```python
X_candidate_desc = np.hstack((
    substrate_descriptors.reshape(1, -1).repeat(X_candidate_desc.shape[0], axis=0), 
    X_candidate_desc
))
```

2. Preparing ID array of the target reaction
* These were used in parallel with the descriptor array to easily point back to the specific chemicals of the selected reactions.
* It can be prepared analogous to how the descriptor array of reaction conditions were generated. The `np.arange()` should be provided for each reaction component. For example, if the reaction condition comprises only catalyst and solvent:
```python
prep_array_of_enumerated_candidates([
    np.arange(1, <number_of_catalysts>), 
    [np.arange(1, <number_of_solvents>)
]])
```
* Note how here we do not need to stack with the substrates, since the target substrate (pair) is fixed.

3. Preparing source models
Using the data in hand, train multiple `RandomForestClassifier`s and collect them in a list. Here, it's important that `max_depth=1` is specified to maximize transferability and `random_state` are different between the models to secure diversity among them. For example,
```python
ATL_source_model_100_trees = []
for j in range(100) :
    rfc = RandomForestClassifier(n_estimators=100, max_depth=1, random_state=42+j, n_jobs=-1)
    rfc.fit(X_ATL_desc_source, y_ATL_source)
    ATL_source_model_100_trees.append(rfc)
```

4. If not the first iteration, preparing target models and combining with previous models
* First prepare the descriptor array of reactions that were conducted, along with the outputs.
* Then, use the `train_target_models()` function in atl_utils.py to prepare the *same number* of target classifiers as the source model.
* Finally, `combine_two_model_list_shuffled` can be used to mix and match previous models with the newly trained target model. 
```python
first_target_models = train_target_models(
    first_round_desc_array, first_round_y, 100, 15, max_depth=1
)
combined_models_after_first_round = combine_two_model_list_shuffled(
    list_of_source_models, first_target_models
)
```

5. Removing the reactions conducted from the previous round
* To ensure that selections are not made from previously conducted experiments, we remove them from the array of candidates using `remove_prev_sampled_rxns()`.
* The arguments are: candidate id array; candidate descriptor array; id array of conducted reactions.
```python
X_candidate_id, X_candidate_desc = remove_prev_sampled_rxns(
    X_candidate_id, X_candidate_desc, first_round_id_array[:, 2:]
)
```
The [:, 2:] is there because substrates are not included in the candidate arrays.

6. Getting the top-N suggestions of reactions to conduct
* First, we let each model vote for N reactions that will likely provide improved reaction outcomes. The results are recorded using `count_num_topN_suggestions()`. 
* Then, then the specific reagents used for the top suggestions can be printed using `print_suggestions()`. Because the structure of conditions for different reactions will be different, 
this function in `atl_utils.py` will need to be modified so that it prints out reagent names appropriately. 

7. Conducting experiments to conclude an iteration of ATL
After conducting the experiments and recording in an excel, steps 4~6 can be repeated.