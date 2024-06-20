import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from dataAnalysis.Constants import *
import torch

def count_cbc_cases(data):
    comp_data = data.query("~(WBC.isnull() & HGB.isnull() & MCV.isnull() & PLT.isnull() & RBC.isnull())",
                           engine='python')
    unique_data = comp_data.drop_duplicates(subset=["Id", "Center"])
    return len(unique_data)


def count_cbc(data):
    comp_data = data.query("~(WBC.isnull() & HGB.isnull() & MCV.isnull() & PLT.isnull() & RBC.isnull())",
                           engine='python')
    return len(comp_data)


class Filter:
    def __init__(self, data):
        unique_data = data.drop_duplicates(subset=["Id", "Center", "Time"], keep=False)
        non_icu_unique_data = unique_data.query("~(Sender.str.contains('ICU')) & ~(~SecToIcu.isnull() & SecToIcu < 0)",
                                                engine='python')
        first_non_icu_unique_data = non_icu_unique_data.query("Episode == 1 ", engine='python')
        complete_first_non_icu_unique_data = first_non_icu_unique_data.query("~(" + " | ".join([i + ".isnull()" for i in FEATURES_IN_TABLE]) +")", engine='python') ## filters all rows with an empty feature value
        sirs_complete_first_non_icu_unique_data = complete_first_non_icu_unique_data.query("Diagnosis != 'SIRS'",
                                                                                           engine='python')
        sirs_complete_first_non_icu_unique_data = \
            sirs_complete_first_non_icu_unique_data.query("(Diagnosis == 'Control') | ((Diagnosis == 'Sepsis') & ("
                                                          "~TargetIcu.isnull() & "
                                                          "TargetIcu.str.contains('MICU')))",
                                                                                           engine='python')
        self.data = sirs_complete_first_non_icu_unique_data
        self.data['Label'] = self.data['Diagnosis']

        control_filter = (self.data["Diagnosis"] == 'Control') | \
                         ((self.data["SecToIcu"] > 3600 * 6) & (
                                     ~self.data["TargetIcu"].isnull() & self.data["TargetIcu"]
                                     .str.contains('MICU', na=False)))
        sepsis_filter = (self.data["Diagnosis"] == 'Sepsis') & \
                        (self.data["SecToIcu"] <= 3600 * 6) & \
                        (self.data["TargetIcu"].str.contains('MICU', na=False))
        self.data.loc[control_filter, "Label"] = "Control"
        self.data.loc[sepsis_filter, "Label"] = "Sepsis"

        self.control_data = self.data.loc[control_filter]
        self.sepsis_data = self.data.loc[sepsis_filter]
        self.resample_data() 

    def get_control_data(self):
        return self.control_data

    def get_sepsis_data(self):
        return self.sepsis_data

    def resample_data(self):
        self.data = self.data.sample(frac=1).reset_index()
        
    def get_data(self):
        return self.data
    
    def get_X(self):
        data = self.get_data()
        data[SEX_COLUMN_NAME] = data[SEX_COLUMN_NAME].astype("category")
        data[SEX_CATEGORY_COLUMN_NAME] = data[SEX_COLUMN_NAME].cat.codes
        return data.loc[:, FEATURES].values
    
    def get_y(self):
        data = self.get_data()
        data[LABEL_COLUMN_NAME] = data[LABEL_COLUMN_NAME].astype('category')
        return (data.loc[:, LABEL_COLUMN_NAME].cat.codes).values
