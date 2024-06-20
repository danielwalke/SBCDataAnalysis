import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
from dataAnalysis.Constants import FEATURES, SEX_CATEGORY_COLUMN_NAME, STEPS, FEATURE_DICT, VARIATION_DF, VARIATION_SMALL_DF
import pandas as pd
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from os.path import exists

def normalize(tensor):
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor).type(torch.float)
    mean = torch.mean(tensor, dim = 0)
    std = torch.std(tensor, dim = 0)
    mean_diff = tensor - mean
    return mean_diff / std

class FeatureImportance:
    def __init__(self, X_train, is_normalize=False, steps = STEPS):
        self.X_train = X_train
        
        self.X_all_fv = pd.read_csv(VARIATION_DF if steps == 20 else VARIATION_SMALL_DF, header=0).to_numpy()[:,1:]
        if is_normalize:
            self.X_all_fv = normalize(self.X_all_fv)
        self.model_input = [self.X_all_fv]
        self.steps = steps
        if (steps == 20 and not exists(VARIATION_DF)) or (steps != 20 and not exists(VARIATION_SMALL_DF)):
            self.write_features_multi_variation()
        
    def feature_variation(self, feature_column):
        feature_column =  feature_column if torch.is_tensor(feature_column) else torch.from_numpy(feature_column)
        q = np.linspace(0, 1, self.steps, endpoint=True)
        # several features have outliers which makes the distribution distrorted, thats why we are using quantiles to check the prediction based on abundance rather than absolute values
        return np.quantile(feature_column, q)
    
    def set_model_input(self, model_input):
        self.model_input = model_input


    def write_features_multi_variation(self):
        X_new = torch.tensor([])
        variations = []
        for feature in FEATURE_DICT:
            if feature== SEX_CATEGORY_COLUMN_NAME:
                continue
            idx = FEATURE_DICT[feature]
            variation = self.feature_variation(self.X_train[:, FEATURE_DICT[feature]])
            variations.append(variation)
        combinations_wo_sex = list(itertools.product(*variations))
        combinations_wo_sex = torch.tensor(combinations_wo_sex)
        tensor_with_male_column = torch.cat((combinations_wo_sex[:, :1], torch.zeros(combinations_wo_sex.shape[0]).unsqueeze(1), combinations_wo_sex[:, 1:]), dim=1)
        tensor_with_female_column = torch.cat((combinations_wo_sex[:, :1], torch.ones(combinations_wo_sex.shape[0]).unsqueeze(1), combinations_wo_sex[:, 1:]), dim=1)
        concatenated_tensor = torch.cat((tensor_with_male_column, tensor_with_female_column), dim=0)
        variation_df = pd.DataFrame(data = concatenated_tensor.numpy(), columns= FEATURES)
        variation_df.to_csv(VARIATION_DF if self.steps == 20 else VARIATION_SMALL_DF)
    
    def get_sepsis_ratios(self, feature, y_pred_log):
        sepsis_ratios = []
        for feature_value in np.unique(self.X_all_fv[:, FEATURE_DICT[feature]]):
            feature_mask = self.X_all_fv[:, FEATURE_DICT[feature]] == feature_value
            uniques, counts = np.unique(y_pred_log[feature_mask], return_counts=True)
            has_both_classes = len(counts) == 2
            if has_both_classes:
                sepsis_ratios.append(counts[1] / (counts[0] + counts[1]))
            else:
                only_class_is_control = uniques[0] == 0
                sepsis_ratios.append(torch.tensor(0) if only_class_is_control else torch.tensor(1))     
        sepsis_ratios = list(map(lambda r: r*100, sepsis_ratios))
        return sepsis_ratios
    
    
    def get_sex_feature_information(self, sepsis_ratios, title = "unknown model"):
        men_diseased_ratio = sepsis_ratios[0]
        women_diseased_ratio = sepsis_ratios[1]
        return f"Ratio of diseased men/women over all CBCs for {title} {str(men_diseased_ratio)}\t{str(women_diseased_ratio)}" 


    def plot_feature_importance(self, model, title = None, write = True):
        y_pred_log = model.predict(*self.model_input)
        y_pred_log =  y_pred_log.cpu() if torch.is_tensor(y_pred_log) else y_pred_log
        feature_sepsis_ratios= []
        sepsis_ratios_stds= []
        for idx, feature in enumerate(FEATURES):
            sepsis_ratios = self.get_sepsis_ratios(feature, y_pred_log)
            if feature == SEX_CATEGORY_COLUMN_NAME:
                print(self.get_sex_feature_information(sepsis_ratios, title))
#                 continue
                diff = sepsis_ratios[1] - sepsis_ratios[0]
                sepsis_ratios = [sepsis_ratios[0] + i/self.steps * diff for i in range(self.steps)]   
            sepsis_ratios_stds.append(np.std(sepsis_ratios))
            feature_sepsis_ratios.append(sepsis_ratios)
        summed_sepsis_ratios = sum(sepsis_ratios_stds)
        feature_variation_df_list = [np.linspace(0, 100, self.steps, endpoint=True)]
        columns = ["Feature variation"]
        legend = []
        for idx, sepsis_ratios in enumerate(feature_sepsis_ratios):
            feature_variation_df_list.append(sepsis_ratios)
            feature_descript = f"{FEATURES[idx]} ({str(round(sepsis_ratios_stds[idx]/summed_sepsis_ratios, 2))})"
            if FEATURES[idx] != SEX_CATEGORY_COLUMN_NAME:
                legend.append(feature_descript)
                plt.plot(np.linspace(0, 100, self.steps, endpoint=True), sepsis_ratios)
            columns.append(feature_descript)
        if write:
            feature_variation_df = pd.DataFrame(data = np.transpose(np.asarray(feature_variation_df_list)), columns = columns)
            feature_variation_df.to_csv(f"{title}.csv")
        plt.xlabel("Feature ratio [%]")
        plt.ylabel("Ratio of Sepsis CBC [%]")
        plt.ylim(0, 100)
        plt.xlim(0, 100)
        plt.title(title)
        plt.grid()
        plt.legend(legend)
        plt.show()