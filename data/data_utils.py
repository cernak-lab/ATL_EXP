import pandas as pd
import numpy as np
import os
from pathlib import Path


SHEET_NAMES = ["NHPI", "Katritzky", "Ni source", "Ligands", "Additives", "Solvents"]
DESC_DIR = os.path.join(Path(__file__).parent.absolute(), "descriptors.xlsx")
ADDITIVE_IDS = np.array(
    [
        [0, 0],  # None
        [5, 1],  # NaCl
        [5, 3],  # NaI
        [6, 1],  # MgCl2
        [6, 2],  # MgBr2
        [7, 1],  # KCl
        [7, 2],  # KBr
        [7, 3],  # KI
        [8, 1],  # ZnCl2
        [9, 9],  # succinimide
        [10, 1],  # TMSCl
        [11, 1],  # TBACl
        [11, 2],  # TBABr
        [11, 3],  # TBAI
    ]
)
ADDITIVE_NAMES = [
    "None",
    "NaCl",
    "NaI",
    "MgCl2",
    "MgBr2",
    "KCl",
    "KBr",
    "KI",
    "ZnCl2",
    "succinimide",
    "TMSCl",
    "TBACl",
    "TBABr",
    "TBAI",
]
COMPONENT_ID_ARRAY = [np.arange(1, 6), np.arange(1, 30), ADDITIVE_IDS, np.arange(1, 10)]

Ni_source_onehot = np.identity(5)
ligand_desc = pd.read_excel(
    "./data/descriptors.xlsx",
    sheet_name="Ligands",
    usecols=[3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 18],
).to_numpy()
additive_ion_onehot = np.zeros((14, 11))
for i, row in enumerate(ADDITIVE_IDS):
    if i != 0:
        for elem in row:
            additive_ion_onehot[i, elem - 1] = 1
solvent_desc = pd.read_excel(
    "./data/descriptors.xlsx", sheet_name="Solvents", usecols=[2, 3, 4, 5, 7, 9]
).to_numpy()
COMPONENT_DESC_ARRAY = [
    Ni_source_onehot,
    ligand_desc,
    additive_ion_onehot,
    solvent_desc,
]


def sheets_to_list_of_dfs(excel_dir, list_of_sheets):
    """Reads an excel file and prepares separate dataframes for each sheet.

    Parameters
    ----------
    excel_dir : str
        Directory of the excel file to read.

    list_of_sheets : list of str
        Names of sheets to extract.

    Returns
    -------
    df_list : list of dataframes
        List of extracted dataframes.
    """
    df_list = []
    for sheet in list_of_sheets:
        df = pd.read_excel(excel_dir, sheet_name=sheet)
        df_list.append(df)
    return df_list


def id_array_to_desc_array(id_array, remove_high_corr_cols=True):
    """Converts an array of compound-IDs into their corresponding array of descriptors.

    Parameters
    ----------
    id_array : np.2darray of shape (n_rxns, n_rxn_components)
        Array of compound id's to convert.

    remove_high_corr_cols : bool
        Whether to prepare an array of highly correlated descriptors within a single reaction component.

    Outputs
    -------
    descriptor_array : np.2darray of shape (n_rxns, n_descriptors)
        Reactions featurized with descriptors.
    """
    desc_dfs = sheets_to_list_of_dfs(DESC_DIR, SHEET_NAMES)

    num_columns = [
        len(desc_dfs[0].columns) - 2,
        len(desc_dfs[1].columns) - 2,
        5,  # one-hot of Ni sources
        len(desc_dfs[3].columns) - 3,
        8,  # one-hot of additive cations
        3,  # one-hot of additive anions
        len(desc_dfs[5].columns)
        - 2,  # will keep MeCN fixed --> not included in descriptor array
    ]

    descriptor_array = np.zeros((id_array.shape[0], sum(num_columns)))

    for i, row in enumerate(id_array):
        for j in range(len(num_columns)):
            if j == 0:
                descriptor_array[i, : num_columns[j]] = desc_dfs[0].iloc[
                    int(row[0]) - 1, 2:
                ]
            elif j in [1, 3, 6]:
                if j < 6:
                    if j == 1:
                        n_subtract = 2
                    elif j == 3:
                        n_subtract = 3
                    descriptor_array[
                        i, sum(num_columns[:j]) : sum(num_columns[: j + 1])
                    ] = desc_dfs[j].iloc[int(row[j]) - 1, n_subtract:]
                else:
                    descriptor_array[i, -1 * num_columns[-1] :] = desc_dfs[j - 1].iloc[
                        int(row[j]) - 1, 2:
                    ]
            elif j == 2:  # nickel source one-hot
                descriptor_array[i, sum(num_columns[:2]) + int(row[j]) - 1] = 1
            else:  # additive one-hot
                if row[j] != 0:
                    descriptor_array[i, sum(num_columns[:4]) + int(row[j]) - 1] = 1
    if remove_high_corr_cols:
        descriptor_array = np.hstack(
            (
                descriptor_array[
                    :, :34
                ],  # 29 descriptors of NHPI, Kat and Ni sources + first 5 ligand descriptors
                descriptor_array[:, 35:37],
                descriptor_array[:, 38:39],
                descriptor_array[:, 40:42],
                descriptor_array[:, 43:60],
                descriptor_array[:, 61:62],
                descriptor_array[:, 63:64],
            )
        )
    return descriptor_array


def prep_array_of_enumerated_candidates(list_of_component_arrays):
    """Prepares an array of all enumerated reaction candidates.
    Not confirmed whether it will work for reactions of more than four components.

    Parameters
    ----------
    list_of_component_arrays : list of ndarray
        Arrays of candidates of each reaction component to be fully enumerated with others in the list.
        Can be id's or descriptor arrays.

    Returns
    -------
    enumerated_candidates : ndarray of shape (n_reactions, n_features)
        All reaction condition candidates.
    """
    num_candidates = [x.shape[0] for x in list_of_component_arrays]
    enumerated_arrays = []
    for i, component_array in enumerate(list_of_component_arrays):
        ndim = component_array.ndim
        if ndim == 1:
            tile_shape = np.prod(num_candidates[:i])
        elif ndim == 2:
            tile_shape = (np.prod(num_candidates[:i]), 1)
        if i == 0:
            enumerated_arrays.append(
                np.repeat(component_array, np.prod(num_candidates[1:]), axis=0)
            )
        elif i < len(list_of_component_arrays) - 1:
            enumerated_arrays.append(
                np.tile(
                    np.repeat(
                        component_array, np.prod(num_candidates[i + 1 :]), axis=0
                    ),
                    tile_shape,
                )
            )
        else:
            enumerated_arrays.append(np.tile(component_array, tile_shape))
        if ndim == 1:
            enumerated_arrays[i] = enumerated_arrays[i].reshape(-1, 1)
    enumerated_candidates = np.hstack(tuple(enumerated_arrays))

    return enumerated_candidates


def prep_full_desc_array_of_candidates(NHPI_row_num, Kat_row_num):
    """The function above does not include fixed portions like NHPI and Katritzky salts.
    This function combines the output of above and the descriptor arrays of NHPI and KAtritzky salt."""
    X_candidate_desc = prep_array_of_enumerated_candidates(COMPONENT_DESC_ARRAY)
    NHPI = pd.read_excel(
        DESC_DIR, sheet_name="NHPI", usecols=list(np.arange(1, 13))
    ).to_numpy()[NHPI_row_num, :]
    Kat = pd.read_excel(
        DESC_DIR, sheet_name="Katritzky", usecols=list(np.arange(1, 13))
    ).to_numpy()[Kat_row_num, :]

    X_candidate_desc = np.hstack(
        (
            np.hstack((NHPI, Kat))
            .reshape(1, -1)
            .repeat(X_candidate_desc.shape[0], axis=0),
            X_candidate_desc,
        )
    )
    return X_candidate_desc


def remove_prev_sampled_rxns(candidate_id_array, candidate_desc_array, prev_id_array):
    """Removes reactions that were sampled previously from the set of candidates.

    Parameters
    ----------
    candidate_id_array : np.2darray of shape (n_reactions, n_rxn_components)
        Enumerated reaction candidates expressed with compound id's

    candidate_desc_array : np.2darray of shape (n_reactions, n_descriptors)
        Enumerated reaction candidates expressed with descriptors.

    prev_id_array : np.2darray of shape (n_reactions_from_prev_batch, n_rxn_components)
        Reactions that were screened in the previous batch, to remove from candidates.

    Output
    ------
    rem_candidate_id : np.2darray of shape (n_remaining_rxns, n_rxn_components)

    rem_candidate_desc : np.2darray of shape (n_reamaining_rxns, n_descriptors)
    """
    inds_to_remove = []
    for row in prev_id_array:
        inds_to_remove.append(np.where((candidate_id_array == row).all(axis=1))[0])
    inds_to_keep = [
        x for x in range(candidate_id_array.shape[0]) if x not in inds_to_remove
    ]
    return candidate_id_array[inds_to_keep, :], candidate_desc_array[inds_to_keep, :]
