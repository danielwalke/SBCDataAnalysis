from dataAnalysis.data.Training import Training
from dataAnalysis.data.Validation import Validation
import pandas as pd
from dataAnalysis.data.Greifswald_Validation import GreifswaldValidation
import numpy as np
from dataAnalysis.scorer.AUROC import area_under_curve


def count_cbc_cases(data):
    comp_data = data.query("~(WBC.isnull() & HGB.isnull() & MCV.isnull() & PLT.isnull() & RBC.isnull())",
                           engine='python')
    unique_data = comp_data.drop_duplicates(subset=["Id", "Center"])
    return len(unique_data)


def count_cbc(data):
    comp_data = data.query("~(WBC.isnull() & HGB.isnull() & MCV.isnull() & PLT.isnull() & RBC.isnull())",
                           engine='python')
    return len(comp_data)


class DataAnalysis:
    def __init__(self, data):
        self.training = Training(data)
        print("Training: ")
        print(f"Assessable data are {count_cbc_cases(self.training.get_data())} cases "
              f"and {count_cbc(self.training.get_data())} CBCs")
        print(f"Control data are {count_cbc_cases(self.training.get_control_data())} cases "
              f"and {count_cbc(self.training.get_control_data())} CBCs")
        print(f"Sepsis data are {count_cbc_cases(self.training.get_sepsis_data())} cases "
              f"and {count_cbc(self.training.get_sepsis_data())} CBCs")
        print(20 * "$")
        print("Testing: ")
        self.validation = Validation(data)
        print(f"Controls: {self.validation.get_control_data().shape[0]},"
              f" Sepsis: {self.validation.get_sepsis_data().shape[0]}")
        print(f"Assessable data are {count_cbc_cases(self.validation.get_data())} cases "
              f"and {count_cbc(self.validation.get_data())} CBCs")
        print(f"Control data are {count_cbc_cases(self.validation.get_control_data())} cases "
              f"and {count_cbc(self.validation.get_control_data())} CBCs")
        print(f"Sepsis data are {count_cbc_cases(self.validation.get_sepsis_data())} cases "
              f"and {count_cbc(self.validation.get_sepsis_data())} CBCs")

        self.greifswald_vaidation = GreifswaldValidation(data)
        print(f"Controls: {self.greifswald_vaidation.get_control_data().shape[0]},"
              f" Sepsis: {self.greifswald_vaidation.get_sepsis_data().shape[0]}")
        print(f"Assessable data are {count_cbc_cases(self.greifswald_vaidation.get_data())} cases "
              f"and {count_cbc(self.greifswald_vaidation.get_data())} CBCs")
        print(f"Control data are {count_cbc_cases(self.validation.get_control_data())} cases "
              f"and {count_cbc(self.greifswald_vaidation.get_control_data())} CBCs")
        print(f"Sepsis data are {count_cbc_cases(self.greifswald_vaidation.get_sepsis_data())} cases "
              f"and {count_cbc(self.greifswald_vaidation.get_sepsis_data())} CBCs")
    
    def get_training_data(self):
        return self.training.get_data()
    
    def get_testing_data(self):
        return self.validation.get_data()
    
    def get_gw_testing_data(self):
        return self.greifswald_vaidation.get_data()
    
    def get_X_train(self):
        return self.training.get_X()
    
    def get_y_train(self):
        return self.training.get_y()
    
    def get_X_test(self):
        return self.validation.get_X()
    
    def get_y_test(self):
        return self.validation.get_y()
    
    def get_X_gw(self):
        return self.greifswald_vaidation.get_X()
    
    def get_y_gw(self):
        return self.greifswald_vaidation.get_y()
