import torch
from sklearn.base import BaseEstimator
from typing import TypedDict
import numpy as np
import numpy
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot  as plt
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance

USER_FUNCTIONS = {
    'sum': lambda origin_features, updated_features, sum_neighbors, mul_neighbors, num_neighbors: sum_neighbors,
    'mean': lambda origin_features, updated_features, sum_neighbors, mul_neighbors, num_neighbors: sum_neighbors / num_neighbors,
    'diff_of_origin_mean': lambda origin_features, updated_features, sum_neighbors, mul_neighbors, num_neighbors: origin_features - sum_neighbors / num_neighbors,
    'diff_of_updated_mean': lambda origin_features, updated_features, sum_neighbors, mul_neighbors, num_neighbors: updated_features - sum_neighbors / num_neighbors,
    'sum_of_origin_mean': lambda origin_features, updated_features, sum_neighbors, mul_neighbors, num_neighbors: origin_features + sum_neighbors / num_neighbors,
    'sum_of_updated_mean': lambda origin_features, updated_features, sum_neighbors, mul_neighbors, num_neighbors: updated_features + sum_neighbors / num_neighbors,
}

def softmax(x):
    e_coef = np.exp(x)
    softmax_coef = e_coef / np.sum(e_coef)
    return softmax_coef

class Framework:    
    
    def __init__(self, user_functions, 
                 hops_list:list[int],
                 clfs:list,
                 multi_target_class:bool =False,
                 gpu_idx:int|None=None,
                 handle_nan:float|None=None,
                attention_configs:list=[],
                scoring_fun = accuracy_score) -> None:
        self.user_functions = user_functions
        self.hops_list:list[int] = hops_list
        self.clfs:list[int] = clfs
        self.trained_clfs = None
        self.gpu_idx:int|None = gpu_idx
        self.handle_nan:float|int|None = handle_nan
        self.attention_configs = attention_configs
        self.multi_target_class = multi_target_class
        self.device:torch.DeviceObjType = torch.device(f"cuda:{str(self.gpu_idx)}") if self.gpu_idx is not None and torch.cuda.is_available() else torch.device("cpu")
        self.num_classes = None
        self.dataset = None
        self.scoring_fun = scoring_fun
        self.multi_out = None
    
    def update_user_function(self):
        if self.user_function in USER_FUNCTIONS:
            self.user_function = USER_FUNCTIONS[self.user_function]
        else:
            raise Exception(f"Only the following string values are valid inputs for the user function: {[key for key in USER_FUNCTIONS]}. You can also specify your own function for aggregatioon.")
            
    def get_features(self,
                     X:torch.FloatTensor,
                     edge_index:torch.LongTensor,
                     mask:torch.BoolTensor,
                    is_training:bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if mask is None:
            mask = torch.ones(X.shape[0]).type(torch.bool)
#         if isinstance(self.user_function, str):
#             self.update_user_function()
        ## To tensor
        X = Framework.get_feature_tensor(X)
        edge_index = Framework.get_edge_index_tensor(edge_index)
        mask = Framework.get_mask_tensor(mask)
        
        ## To device
        X = self.shift_tensor_to_device(X)
        edge_index = self.shift_tensor_to_device(edge_index)
        mask = self.shift_tensor_to_device(mask)
        
        aggregated_train_features_list = []
        ## Aggregate
        for hop_idx in range(len(self.hops_list)):
            neighbor_features = self.aggregate(X, edge_index, hop_idx, is_training)
            aggregated_train_features_list.append(neighbor_features[mask])
        return aggregated_train_features_list


    def feature_aggregations(self, features, target, source_lift):
        summed_neighbors = torch.zeros_like(features, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features.shape[1], 1).t(), source_lift, reduce="sum", include_self = False)
        multiplied_neighbors = torch.ones_like(features, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features.shape[1], 1).t(), source_lift, reduce="prod", include_self = False)
        mean_neighbors = torch.zeros_like(features, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features.shape[1], 1).t(), source_lift, reduce="mean", include_self = False)
        max_neighbors = torch.zeros_like(features, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features.shape[1], 1).t(), source_lift, reduce="amax", include_self = False)
        min_neighbors = torch.zeros_like(features, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features.shape[1], 1).t(), source_lift, reduce="amin", include_self = False)
        return summed_neighbors, multiplied_neighbors, mean_neighbors, max_neighbors, min_neighbors
    
    def aggregate(self, X:torch.FloatTensor, edge_index:torch.LongTensor,hop_idx, is_training:bool=False) -> torch.FloatTensor: 
        original_features = X
        features_for_aggregation:torch.FloatTensor = torch.clone(X)
        hops_list = self.hops_list[hop_idx]
        for i, hop in enumerate(range(hops_list)):
            if self.attention_configs[hop_idx] and self.attention_configs[hop_idx]["inter_layer_normalize"]:
                features_for_aggregation = torch.nn.functional.normalize(features_for_aggregation, dim = 0)
            source_lift = features_for_aggregation.index_select(0, edge_index[0])
            source_origin_lift = original_features.index_select(0, edge_index[0])
            target = edge_index[1]
            
            if self.attention_configs[hop_idx] and self.attention_configs[hop_idx]["use_pseudo_attention"]:
                source_lift = self.apply_attention_mechanism(source_lift, features_for_aggregation, target,self.attention_configs[hop_idx], is_training)

            summed_neighbors, multiplied_neighbors, mean_neighbors, max_neighbors, min_neighbors = self.feature_aggregations(features_for_aggregation, target, source_lift)
            summed_origin_neighbors, multiplied_origin_neighbors, mean_origin_neighbors, max_origin_neighbors, min_origin_neighbors = self.feature_aggregations(original_features, target, source_origin_lift)
            # summed_neighbors = torch.zeros_like(features_for_aggregation, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features_for_aggregation.shape[1], 1).t(), source_lift, reduce="sum", include_self = False)
            # multiplied_neighbors = torch.ones_like(features_for_aggregation, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features_for_aggregation.shape[1], 1).t(), source_lift, reduce="prod", include_self = False)
            # mean_neighbors = torch.zeros_like(features_for_aggregation, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features_for_aggregation.shape[1], 1).t(), source_lift, reduce="mean", include_self = False)
            # max_neighbors = torch.zeros_like(features_for_aggregation, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features_for_aggregation.shape[1], 1).t(), source_lift, reduce="amax", include_self = False)
            # min_neighbors = torch.zeros_like(features_for_aggregation, device=self.device).scatter_reduce(0, target.unsqueeze(0).repeat(features_for_aggregation.shape[1], 1).t(), source_lift, reduce="amin", include_self = False)

            num_source_neighbors = torch.zeros(features_for_aggregation.shape[0], dtype=torch.float, device=self.device)
            num_source_neighbors.scatter_reduce(0, target, torch.ones_like(target, dtype=torch.float, device=self.device), reduce="sum", include_self = False)
            num_source_neighbors = num_source_neighbors.unsqueeze(-1)

            user_function = self.user_functions[hop_idx]
            updated_features = features_for_aggregation ## just renaming so that the key in the user function is clear
            user_function_kwargs = {
                                'original_features':original_features,
                                'updated_features':updated_features,
                                'summed_neighbors':summed_neighbors,
                                'multiplied_neighbors':multiplied_neighbors,
                                'mean_neighbors':mean_neighbors,
                                'max_neighbors':max_neighbors,
                                'min_neighbors':min_neighbors,
                                'summed_origin_neighbors':summed_origin_neighbors,
                                'multiplied_origin_neighbors':multiplied_origin_neighbors,
                                'mean_origin_neighbors':mean_origin_neighbors,
                                'max_origin_neighbors':max_origin_neighbors,
                                'min_origin_neighbors':min_origin_neighbors,
                                'num_source_neighbors':num_source_neighbors,
                                'hop':hop}
            out = user_function(user_function_kwargs)
            
            if self.handle_nan is not None:
                out = torch.nan_to_num(out, nan=self.handle_nan)
            features_for_aggregation = out
        return features_for_aggregation
    
    def apply_attention_mechanism(self, source_lift:torch.FloatTensor,
                                  features_for_aggregation:torch.FloatTensor,
                                  target:torch.LongTensor,
                                  attention_config,
                                 is_training:bool = False) -> torch.FloatTensor:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        score = cos(source_lift, features_for_aggregation.index_select(0, target))
        dropout_tens = None
        
        origin_scores = torch.clone(score)
        if attention_config["cosine_eps"]:
            score[score < attention_config["cosine_eps"]] = -torch.inf
        if attention_config["dropout_attn"] is not None and is_training:
            dropout_tens = torch.FloatTensor(score.shape[0]).uniform_(0, 1)
            score[dropout_tens < attention_config["dropout_attn"]] = -torch.inf
        exp_score = torch.exp(score)
        summed_exp_score = torch.zeros_like(exp_score).scatter(0, target,exp_score, reduce="add")
        target_lifted_summed_exp_score = summed_exp_score.index_select(0, target)
        normalized_scores = exp_score / target_lifted_summed_exp_score
        source_lift = normalized_scores.unsqueeze(1) * source_lift
        return source_lift
    
    def fit(self,
            X_train:torch.FloatTensor,
            edge_index:torch.LongTensor,
            y_train:torch.LongTensor,
            train_mask:torch.BoolTensor|None,
            kwargs_fit_list = None,
            transform_kwargs_fit = None,
            kwargs_multi_clf_list = None
            ) -> BaseEstimator:   
        if train_mask is None:
            train_mask = torch.ones(X_train.shape[0]).type(torch.bool)
            
        y_train = Framework.get_label_tensor(y_train)
        y_train = y_train[train_mask]
        self.num_classes = len(y_train.unique(return_counts = True)[0])
        self.multi_out = y_train.shape[-1]
        
        
        self.validate_input()
        
        aggregated_train_features_list = self.get_features(X_train, edge_index, train_mask, True)  
        
        trained_clfs = []
        for i, aggregated_train_features in enumerate(aggregated_train_features_list):
            clf = clone(self.clfs[i])
            if self.multi_target_class:
                kwargs_multi_clf = kwargs_multi_clf_list[i] if kwargs_multi_clf_list and len(kwargs_multi_clf_list)>i is not None else {}
                clf = MultiOutputClassifier(clf, **kwargs_multi_clf)
            kwargs = kwargs_fit_list[i] if kwargs_fit_list and len(kwargs_fit_list)>i is not None else {}
            transformed_kwargs = transform_kwargs_fit(self, kwargs, i) if transform_kwargs_fit is not None else kwargs
            clf.fit(aggregated_train_features.cpu().numpy(), y_train,**transformed_kwargs)
            trained_clfs.append(clf)
        self.trained_clfs = trained_clfs
        return trained_clfs    
    
    def predict_proba(self, X_test:torch.FloatTensor,
                      edge_index:torch.LongTensor,
                      test_mask:torch.BoolTensor|None,
                      weights=None,
                     kwargs_list = None):  
        if test_mask is None:
            test_mask = torch.ones(X_test.shape[0]).type(torch.bool)
        aggregated_test_features_list = self.get_features(X_test, edge_index, test_mask)
        
        pred_probas = []
        for i, clf in enumerate(self.trained_clfs):
            aggregated_test_features = aggregated_test_features_list[i]
            kwargs = kwargs_list[i] if kwargs_list is not None else {}
            pred_proba = clf.predict_proba(aggregated_test_features.cpu().numpy(),**kwargs) if kwargs else clf.predict_proba(aggregated_test_features.cpu().numpy())
            pred_probas.append(pred_proba)
        final_pred_proba = np.average(np.asarray(pred_probas), weights=weights, axis=0)
        if self.multi_target_class:
            return np.transpose(final_pred_proba, axes = [1,0,2])
        return final_pred_proba
        
    
    def predict(self,
                X_test:torch.FloatTensor,
                edge_index:torch.LongTensor,
                test_mask:torch.BoolTensor|None,
                 weights=None,
                     kwargs_list = None):
        return self.predict_proba(X_test, edge_index, test_mask, weights, kwargs_list).argmax(-1)
        

    def validate_input(self):
        pass
            
    @staticmethod
    def get_feature_tensor(X:torch.FloatTensor) -> torch.FloatTensor|None:
        if not torch.is_tensor(X):
            try:
                return torch.from_numpy(X).type(torch.float)
            except:
                raise Exception("Features input X must be numpy array or torch tensor!")
                return None 
        return X
    
    @staticmethod
    def get_label_tensor(y:torch.LongTensor) -> torch.LongTensor|None:
        if not torch.is_tensor(y):
            try:
                return torch.from_numpy(y).type(torch.long)
            except:
                raise Exception("Label input y must be numpy array or torch tensor!")
                return None
        return y
    
    @staticmethod
    def get_mask_tensor(mask:torch.BoolTensor) -> torch.BoolTensor|None:
        if not torch.is_tensor(mask):
            try:
                return torch.from_numpy(mask).type(torch.bool)
            except:
                raise Exception("Input mask must be numpy array or torch tensor!")
                return None
        return mask
            
    @staticmethod
    def get_edge_index_tensor(edge_index:torch.LongTensor) -> torch.LongTensor|None:
        if not torch.is_tensor(edge_index):
            try:
                edge_index =  torch.from_numpy(edge_index).type(torch.long)
                Framework.validate_edge_index(edge_index)
                return edge_index
            except:
                raise Exception("Edge index must be numpy array or torch tensor")
                return None
        return edge_index
    
    @staticmethod
    def validate_edge_index(edge_index:torch.LongTensor) -> None:
        if edge_index.shape[0] != 2:
            raise Exception("Edge index must have the shape 2 x NumberOfEdges")
            # TODO: check max edge index and shape of features
    
    def shift_tensor_to_device(self,
                               t:torch.FloatTensor) -> torch.FloatTensor:
        if self.gpu_idx is not None:
            return t.to(self.device) 
        return t
    
    def validate_grid_input(self, grid_params):
        if len(grid_params) != 1 and self.use_feature_based_aggregation:
            raise Exception("You need to provide grid parameter for the classifier!")
        if len(grid_params) != 2 and not self.use_feature_based_aggregation:
            raise Exception("You need to provide two grid parameter, one for each classifier!")
        return

    def set_dataset(self, dataset):
        ## dict with features in key "X", edge index in key "edge_index" and labels in key "y", mask under "mask"
        self.dataset = dataset

    def feature_importance_per_class(self, class_idx, n_repeats = 1):
        framework = self
        dataset = self.dataset
        if self.multi_target_class:
            is_linear_clfs = all([hasattr(framework.trained_clfs[i].estimators_[0], 'coef_')  for i in range(len(framework.trained_clfs))])
            if class_idx:
                return np.mean([softmax(np.abs(framework.trained_clfs[i].estimators_[class_idx].coef_)) for i in range(len(framework.trained_clfs))], axis = 0)
            return np.mean([softmax(np.abs(framework.trained_clfs[i].estimators_[:].coef_)) for i in range(len(framework.trained_clfs))], axis = 0)
        if not self.multi_target_class:
            if class_idx:
                return np.mean([softmax(np.abs(framework.trained_clfs[i].coef_[class_idx])) for i in range(len(framework.trained_clfs))], axis = 0)
            return np.mean([softmax(np.abs(framework.trained_clfs[i].coef_)) for i in range(len(framework.trained_clfs))], axis = 0)
    
    def feature_importance(self, n_repeats = 10):
        framework = self
        num_classes = self.num_classes if not self.multi_target_class else self.multi_out
        if not num_classes: raise Exception("Not fitted yet")
        if self.multi_target_class:
            is_tree_clfs = all([hasattr(framework.trained_clfs[i].estimators_[0], 'feature_importances_') for i in range(len(framework.trained_clfs))])
            if is_tree_clfs:
                return np.mean([np.mean([framework.trained_clfs[i].estimators_[class_idx].feature_importances_ for i in range(len(framework.trained_clfs))], axis = 0) for class_idx in range(num_classes)], axis = 0)
            is_linear_clfs = all([hasattr(framework.trained_clfs[i].estimators_[0], 'coef_')  for i in range(len(framework.trained_clfs))])
            if is_linear_clfs:
                return np.mean([np.mean([softmax(np.abs(framework.trained_clfs[i].estimators_[class_idx].coef_[0])) for i in range(len(framework.trained_clfs))], axis = 0) for class_idx in range(num_classes)], axis = 0)
            if self.dataset is None: raise Exception("Dataset have to be set for calculating feature importance in non-tree-based or non-linear Classifiers")
            return np.mean([np.mean([permutation_importance(framework.trained_clfs[i].estimators_[class_idx], 
                       framework.get_features(self.dataset["X"],
                                              self.dataset["edge_index"],
                                              self.dataset["mask"])[i].cpu(),
                        self.dataset["y"][self.dataset["mask"]][:, class_idx],
                        n_repeats=10,
                        random_state=0)["importances_mean"] for class_idx in range(num_classes)], axis = 0) for i in range(len(framework.trained_clfs))], axis = 0)
        if not self.multi_target_class:
            is_tree_clfs = all([hasattr(framework.trained_clfs[i], "feature_importances_") for i in range(len(framework.trained_clfs))])
            if is_tree_clfs:
                return np.mean([framework.trained_clfs[i].feature_importances_ for i in range(len(framework.trained_clfs))], axis = 0)
            is_linear_clfs = all([hasattr(framework.trained_clfs[i], "coef_") for i in range(len(framework.trained_clfs))])
            if is_linear_clfs:
                return np.mean([framework.trained_clfs[i].coef_[0] for i in range(len(framework.trained_clfs))], axis = 0)
        if self.dataset is None: raise Exception("Dataset dict({'X':features, 'y':labels, 'edge_index':edge_index, 'mask':boolean-mask}) have to be set (set_dataset) for calculating feature importance in non-tree-based or non-linear Classifiers")
        return np.mean([permutation_importance(framework.trained_clfs[i], framework.get_features(self.dataset["X"],
                                              self.dataset["edge_index"],
                                              self.dataset["mask"])[i].cpu(), self.dataset["y"][self.dataset["mask"]],
                            n_repeats=n_repeats,
                          random_state=0)["importances_mean"]  for i in range(len(framework.trained_clfs))], axis = 0)

    def plot_feature_importances(self, mark_top_n_peaks = 3, which_grid = "both", file_name = None):
        y = self.feature_importance()
        x = np.arange(y.shape[0])
        peaks_idx = y.argsort()[::-1][:mark_top_n_peaks]
        plt.bar(x,y)
        plt.ylabel("Relative Importance")
        plt.xlabel("Features")
        plt.scatter(x[peaks_idx], y[peaks_idx], c='red', marker='o')
        for i, peak in enumerate(peaks_idx):
            plt.annotate(f'{x[peak]:.0f}', (x[peak], y[peak]), textcoords="offset points", xytext=(0,10), ha='center')
        if which_grid:
            plt.grid(visible=True, which=which_grid)
        plt.show()
        if file_name: plt.savefig(f"{file_name}.png")

    def plot_tsne(self, X, edge_index, y, mask = None, label_to_color_map = None):
        mask = torch.ones(X.shape[0]).type(torch.bool) if mask is None else mask
        scores = self.predict_proba(X, edge_index, mask)
        node_labels = y[mask].cpu().numpy()
        if self.multi_target_class or (self.num_classes == 2 and score.shape[-1] > 2): raise Exception("Currently not supported for multi class prediction")

        num_classes = self.num_classes         
        t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(scores)
        
        fig = plt.figure(figsize=(12,8), dpi=80)  # otherwise plots are really small in Jupyter Notebook
        label_to_color_map = {i: (np.random.random(), np.random.random(), np.random.random()) for i in range(num_classes)} if label_to_color_map is None else label_to_color_map
        for class_id in range(num_classes):
            plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=20, color=label_to_color_map[class_id], edgecolors='black', linewidths=0.2)
        plt.legend(label_to_color_map.keys())
        plt.show()