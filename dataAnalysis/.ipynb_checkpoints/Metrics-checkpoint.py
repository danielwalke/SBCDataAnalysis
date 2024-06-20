from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from torchmetrics import AUROC
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def custom_metrics(y_true, y_pred, y_score, name):
        mcc = matthews_corrcoef(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_binary = f1_score(y_true, y_pred, average='binary')
        auroc = roc_auc_score(y_true, y_score)
        auc_precision_recall = average_precision_score(y_true, y_score)
        return {
            "Name": name,
            "MCC": mcc,
            "F1-Micro": f1_micro,
            "F1-Macro": f1_macro,
            "F1-Binary": f1_binary,
            "AUROC": auroc,
            "AUPRC": auc_precision_recall,
        }

class Evaluation:
    def __init__(self, y_test, y_gw_test, X_test, X_gw_test, y_mimic = None, X_mimic = None):
        self.y_test = y_test
        self.y_gw_test = y_gw_test
        self.X_test = X_test
        self.X_gw_test = X_gw_test
        self.y_mimic = y_mimic
        self.X_mimic = X_mimic
        self.test_args = None
        self.gw_args = None
        self.mimic_args = None
    
    def set_gw_args(self, gw_args):
        self.gw_args = gw_args
        
    def set_test_args(self, test_args):
        self.test_args = test_args

    def set_mimic_args(self, mimic_args):
        self.mimic_args = mimic_args

    def get_df_metrics(self, model):
        test_pred = model.predict(*self.test_args) if self.test_args is not None else model.predict(self.X_test)
        test_pred_proba = model.predict_proba(*self.test_args) if self.test_args is not None else model.predict_proba(self.X_test)
        gw_pred = model.predict(*self.gw_args) if self.gw_args is not None else model.predict(self.X_gw_test)
        gw_pred_proba = model.predict_proba(*self.gw_args) if self.gw_args is not None else model.predict_proba(self.X_gw_test)
        y_mimic = None
        if self.X_mimic is not None:
            mimic_pred = model.predict(*self.mimic_args) if self.mimic_args is not None else model.predict(self.X_mimic)
            mimic_pred_proba = model.predict_proba(*self.mimic_args) if self.mimic_args is not None else model.predict_proba(self.X_mimic)
            
            y_mimic = Evaluation.create_y_dict(mimic_pred, mimic_pred_proba, self.y_mimic)
            y_mimic["name"] = "MIMIC-IV-v1"
        
        y_leipzig = Evaluation.create_y_dict(test_pred, test_pred_proba, self.y_test)
        y_leipzig["name"] = "Leipzig"
        y_gw = Evaluation.create_y_dict(gw_pred, gw_pred_proba, self.y_gw_test)
        y_gw["name"] = "Greifswald"
        return Evaluation.get_df_metrics_from_pred(y_leipzig, y_gw) if y_mimic is None else Evaluation.get_df_metrics_from_pred(y_leipzig, y_gw, y_mimic)
        
    def plot_confusion_matrix(self, model):
        y_pred = model.predict(*self.test_args) if self.test_args is not None else model.predict(self.X_test)
        Evaluation.plot_confusion_matrix_from_pred(y_pred, self.y_test)
        
    @staticmethod
    def create_y_dict(test_pred, test_pred_proba, y_test):
        return {
            "test_pred": test_pred,
            "test_pred_proba": test_pred_proba,
            "y_test": y_test
        }
        
    @staticmethod
    def plot_confusion_matrix_from_pred(y_pred, y_test):
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Control", "Sepsis"])
        cm_display.plot()
        
    @staticmethod
    def get_df_metrics_from_pred(*args):
        data = []
        for i, arg in enumerate(args):
            if "name" not in arg:
                arg["name"] = ["Leipzig", "Greifswald"][i]
            test_pred, test_pred_proba, y_test = arg["test_pred"], arg["test_pred_proba"], arg["y_test"]
    
            test_metrics = custom_metrics(y_test, test_pred, test_pred_proba[:, 1], arg["name"])

            data.append([test_metrics[key] for key in test_metrics])
        COLUMNS = [key for key in test_metrics] #["NAME", "MCC", "F1-Micro", "F1-Macro", "F1-Binary", "AUROC", "AUPRC"]
        df = pd.DataFrame(data=data, columns=COLUMNS)
        return df