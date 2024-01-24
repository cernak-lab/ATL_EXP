# Prospective Active Transfer Learning on the C–C Coupling of Amines and Carboxylic Acids
Code to reproduce the paper.

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