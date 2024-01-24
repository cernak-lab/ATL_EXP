import numpy as np
import pandas as pd
import joblib
from data_utils import *

SHEET_NAMES = ["NHPI", "Katritzky", "Ni source", "Ligands", "Additives", "Solvents"]
DESC_DIR = "./data/descriptors.xlsx"
RXN_DIR = "./data/reaction_data.xlsx"


def main():
    desc_df = sheets_to_list_of_dfs(DESC_DIR, SHEET_NAMES)

    id_df = sheets_to_list_of_dfs(RXN_DIR, ["Source"])[0]
    id_array = id_df.iloc[:, 1:-1].to_numpy()

    descriptor_array = id_array_to_desc_array(
        id_array, True
    )  # False retains all descriptors regardless of correlation

    joblib.dump(descriptor_array, "./joblib_files/X_desc.joblib")
    joblib.dump(id_array, "./joblib_files/X_id.joblib")

    y_raw = id_df.iloc[:, -1].to_numpy()
    # Coupling of BocPro + BnKat --> Threshold : 50% LC yield
    Pro_Bn_pos_inds = np.where(y_raw[:71] >= 50)[0]

    # Coupling of BocPro + IndanKat --> Threshold : Prod/IS = 9
    Pro_Indan_pos_inds = np.where(y_raw[71:83] >= 9)[0] + 71

    # Coupling of Ala + IndanKat --> Threshold : Prod/IS = 3
    Ala_Indan_pos_inds = np.where(y_raw[83:] > 7)[0] + 83

    y_binary = np.zeros(len(id_array))
    y_binary[Pro_Bn_pos_inds] = 1
    y_binary[Pro_Indan_pos_inds] = 1
    y_binary[Ala_Indan_pos_inds] = 1

    assert len(y_binary) == id_array.shape[0]
    joblib.dump(y_binary, "joblib_files/y.joblib")


if __name__ == "__main__":
    main()
