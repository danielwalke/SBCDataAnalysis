import numpy as np
import torch
from sklearn.inspection import partial_dependence
from torch_geometric.data import Data
from scipy.stats.mstats import mquantiles
import os
import pandas as pd
from tqdm.notebook import tqdm

SEX_COLUMN_NAME = "Sex"
AGE_COLUMN_NAME = "Age"
HGB_COLUMN_NAME = "HGB"
WBC_COLUMN_NAME = "WBC"
RBC_COLUMN_NAME = "RBC"
MCV_COLUMN_NAME = "MCV"
PLT_COLUMN_NAME = "PLT"
FEATURES = [AGE_COLUMN_NAME, SEX_COLUMN_NAME, HGB_COLUMN_NAME, WBC_COLUMN_NAME, RBC_COLUMN_NAME, MCV_COLUMN_NAME, PLT_COLUMN_NAME]

class PartialDependence:
    def __init__(self, model, X, edge_index = None, resolution = 100, extremes = [0.05, 0.95], scaler = None, create_graph = None):
        self.model = model
        self.X = X
        self.resolution = resolution
        self.extremes = extremes
        self.edge_index = edge_index
        self.grid = dict()
        self.pdps = dict()
        self.scaler = scaler
        self.create_graph = create_graph

    def create_grid(self, feature_idx):
        X, extremes, resolution = self.X, self.extremes, self.resolution
        if np.unique(X[:, feature_idx]).shape[0] <= resolution:
            return np.unique(X[:, feature_idx])
        emp_percentiles = mquantiles(
                    X[:, feature_idx], prob=extremes, axis=0
                )
        axis = np.linspace(
                    emp_percentiles[0],
                    emp_percentiles[1],
                    num=resolution,
                    endpoint=True,
                )
        return axis

    def partial_dependence_values(self, feature_idx):
        grid = self.create_grid(feature_idx)
        self.grid[feature_idx] = grid
        X, model = self.X, self.model
        edge_index = self.edge_index
        avg_pred_probas = np.zeros((grid.shape[0]))
        for i, grid_item in enumerate(grid):
            X_clone = torch.clone(self.X).numpy() if torch.is_tensor(self.X) else np.copy(X)
            X_clone[:, feature_idx] = grid_item
            if self.scaler:
                X_clone = self.scaler.transform(X_clone)
            X_clone = torch.from_numpy(X_clone).type(torch.float) if torch.is_tensor(self.X) else X_clone
            data = self.create_graph(X_clone, edge_index=edge_index) if self.create_graph is not None else X_clone
            mask = torch.ones(X_clone.shape[0]).type(torch.bool) if self.create_graph is not None else None
            pred_proba = model.predict_proba(data, mask) if mask is not None  else model.predict_proba(data)
            avg_pred_probas[i] = torch.mean(pred_proba[:, 1]) if torch.is_tensor(self.X) else np.mean(pred_proba[:, 1])
        self.pdps[feature_idx] = avg_pred_probas
        return avg_pred_probas

    def close_sklearn(self, feature_idx):
        if feature_idx not in self.grid:
            self.partial_dependence_values(feature_idx)
        results = partial_dependence(self.model, self.X, [feature_idx])
        assert np.allclose(self.grid[feature_idx], results["grid_values"][0]), "Scikit-learn grid differs"
        assert np.allclose(self.pdps[feature_idx], results["average"][0]), "Scikit-learn results differ"
        print("\x1b[32mSuccess\x1b[0m")
        return True

    def feature_importance(self, feature_idx):
        if len(self.pdps) != self.X.shape[-1]:
            raise Exception("Not all partial dependencies calculated from feature indices in the dataset.")
        raw_fis = np.array([self.pdps[i].var() for i in range(self.X.shape[-1])])
        return  raw_fis/raw_fis.sum()

    def get_all_partial_dependence_values(self):
        for feature_idx in tqdm(range(self.X.shape[-1])):
            if feature_idx in self.pdps:
                continue
            self.partial_dependence_values(feature_idx)
        return self.pdps

    def write_partial_dependence_values(self, class_name = None):
        
        pdps = self.get_all_partial_dependence_values()
        folder_name = class_name if class_name else self.model.__class__.__name__
        folder_name = f"{folder_name}_partial_dependence"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        for feature_idx in pdps:
            df = pd.DataFrame({'grid_values': self.grid[feature_idx], 'average_pred': pdps[feature_idx]})
            df.to_csv(os.path.join(folder_name, f"{FEATURES[feature_idx]}.csv"),index = False)
        
        