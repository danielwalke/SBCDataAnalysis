from hyperopt import fmin, tpe, hp,STATUS_OK, SparkTrials, space_eval 
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from EnsembleFramework import Framework
from torch.nn.functional import normalize
from sklearn.multioutput import MultiOutputClassifier
import torch 
from torch import nn
import torch

def upd_user_function(kwargs):
    return  nn.functional.normalize(kwargs["updated_features"] + kwargs["summed_neighbors"], p = 2.0, dim = -1)
# DEF_ATTENTION_CONFIGS= [None,{'inter_layer_normalize': False,
#                      'use_pseudo_attention':True,
#                      'cosine_eps':.01,
#                      'dropout_attn': None}, 
#                      {'inter_layer_normalize': True,
#                      'use_pseudo_attention':True,
#                      'cosine_eps':.01,
#                      'dropout_attn': None},
#                      {'inter_layer_normalize': True,
#                      'use_pseudo_attention':True,
#                      'cosine_eps':.001,
#                      'dropout_attn': None}]
DEF_ATTENTION_CONFIGS= [{'inter_layer_normalize': False,
                     'use_pseudo_attention':True,
                     'cosine_eps':.01,
                     'dropout_attn': None}]
DEF_HOPS = [[0, 2], [0, 3], [0, 6]]
DEF_MAX_EVALS = 1000
def norm_user_function(kwargs):
    return  normalize(kwargs["original_features"] + kwargs["summed_neighbors"], p=2.0, dim = 1)
    
def user_function(kwargs):
    return  kwargs["original_features"] + kwargs["summed_neighbors"]
    
DEF_USER_FUNCTIONS = [user_function] #upd_user_function,norm_user_function

class Data():
    def __init__(self, X, y, edge_index):
        self.X = X
        self.y = y
        self.edge_index = edge_index
        self.train = None
        self.val = None
        self.test = None
        # self.X_train = None
        # self.X_val = None
        # self.X_test = None
        # self.y_train = None
        # self.y_val = None
        # self.y_test = None
    
    def set_train(self, train):
        self.train = train

    def set_test(self, test):
        self.test = test

    def set_val(self, val):
        self.val = val

    # def set_X_train(self, X):
    #     self.X_train = X

    # def set_X_val(self, X):
    #     self.X_val = X

    # def set_X_test(self, X):
    #     self.X_test = X

class SparkTune():
    def __init__(self, clf,user_function,hops,attention_config, auto_search):
        self.clf = clf
        self.auto_search = auto_search
        self.user_function = user_function
        self.hops = hops
        self.attention_config = attention_config
        
    def objective(self, params):
        model = self.clf(**params)
        auto_search = self.auto_search
        framework = Framework([self.user_function  for hop in self.hops], 
                         hops_list=[hop for hop in self.hops],
                         clfs=[model for hop in self.hops],
                         gpu_idx=0,
                         handle_nan=0.0,
                        attention_configs=[self.attention_config for hop in self.hops], multi_target_class=auto_search.multi_target_class)
        score = None
        if auto_search.is_transductive:
            framework.fit(auto_search.data.X, auto_search.data.edge_index,
                          auto_search.data.y, auto_search.data.train, kwargs_multi_clf_list = [{"n_jobs":11}])
            y_pred = framework.predict_proba(auto_search.data.X, auto_search.data.edge_index, auto_search.data.val)[:,1]
            score = auto_search.pred_metric(auto_search.data.y[auto_search.data.val],
                                            y_pred,
                                            **auto_search.pred_metric_kwargs)
        if not auto_search.is_transductive:
            framework.fit(auto_search.train_data.X, auto_search.train_data.edge_index,
                          auto_search.train_data.y, torch.ones(auto_search.train_data.X.shape[0]).type(torch.bool), kwargs_multi_clf_list = [{"n_jobs":11}])
            y_pred = framework.predict_proba(auto_search.val_data.X, auto_search.val_data.edge_index,
                                       torch.ones(auto_search.val_data.X.shape[0]).type(torch.bool))[:,1]
            score = auto_search.pred_metric(auto_search.val_data.y,
                                            y_pred,
                                            **auto_search.pred_metric_kwargs)
        return {'loss': -score, 'status': STATUS_OK}
    
    def search(self, space):
        spark_trials = SparkTrials(parallelism = self.auto_search.parallelism)
        best_params = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.auto_search.max_evals, trials=spark_trials, verbose = False)
        return best_params


class AutoSearch:
    
    def __init__(self, data_dict, max_evals = 200, multi_target_class = False, pred_metric= accuracy_score, pred_metric_kwargs = {}, is_transductive = True, parallelism = 3):
        self.data_dict = data_dict
        self.max_evals = max_evals
        self.multi_target_class = multi_target_class
        self.pred_metric = pred_metric
        self.pred_metric_kwargs = pred_metric_kwargs
        self.is_transductive = is_transductive
        self.data:Data = None
        self.train_data:Data = None
        self.val_data:Data = None
        self.test_data:Data = None
        self.parallelism = parallelism


    def parse_data(self):
        dataset = self.data_dict
        if self.is_transductive:
            self.data = Data(dataset["X"], dataset["y"], dataset["edge_index"])
            self.data.set_test(dataset["test"])
            self.data.set_val(dataset["val"])
            self.data.set_train(dataset["train"])
        if not self.is_transductive:
            self.train_data = Data(dataset["X_train"], dataset["y_train"], dataset["edge_index_train"])
            self.val_data = Data(dataset["X_val"], dataset["y_val"], dataset["edge_index_val"])
            self.test_data = Data(dataset["X_test"], dataset["y_test"], dataset["edge_index_test"])

    def search_hop_clf_attention_config(self, hops, clf, user_function, attention_config, space):
        self.parse_data()
        
        sparkTune = SparkTune(clf,user_function,hops,attention_config, self)
        params = sparkTune.search(space)
        params = space_eval(space, params) ## index choices to original choices
        
        model = clf(**params)
        framework = Framework([user_function for hop in hops], 
                         hops_list=[hop for hop in hops],
                         clfs=[model for hop in hops],
                         gpu_idx=0,
                         handle_nan=0.0,
                        attention_configs=[attention_config  for hop in hops], multi_target_class=self.multi_target_class)
        if self.is_transductive:
            framework.fit(self.data.X, self.data.edge_index, self.data.y, self.data.train, kwargs_multi_clf_list = [{"n_jobs":11}])
        if not self.is_transductive:
            framework.fit(self.train_data.X, self.train_data.edge_index, self.train_data.y, torch.ones(self.train_data.X.shape[0]).type(torch.bool), kwargs_multi_clf_list = [{"n_jobs":11}])

        train_acc, val_acc, test_acc = None, None, None
        if self.is_transductive:
            val_pred = framework.predict_proba(self.data.X, self.data.edge_index,
                                        self.data.val)[:,1]
            train_pred = framework.predict_proba(self.data.X, self.data.edge_index,
                                        self.data.train)[:,1]
            test_pred = framework.predict_proba(self.data.X, self.data.edge_index,
                                        self.data.test)[:,1]
            train_acc = self.pred_metric(self.data.y[self.data.train], train_pred, **self.pred_metric_kwargs)
            val_acc = self.pred_metric(self.data.y[self.data.val], val_pred, **self.pred_metric_kwargs)
            test_acc = self.pred_metric(self.data.y[self.data.test], test_pred, **self.pred_metric_kwargs)
        if not self.is_transductive:
            val_pred = framework.predict_proba(self.val_data.X, self.val_data.edge_index,
                                           torch.ones(self.val_data.X.shape[0]).type(torch.bool))[:,1]
            train_pred = framework.predict_proba(self.train_data.X, self.train_data.edge_index,
                                           torch.ones(self.train_data.X.shape[0]).type(torch.bool))[:,1]
            test_pred = framework.predict_proba(self.test_data.X, self.test_data.edge_index,
                                           torch.ones(self.test_data.X.shape[0]).type(torch.bool))[:,1]
            train_acc = self.pred_metric(self.train_data.y, train_pred, **self.pred_metric_kwargs)
            val_acc = self.pred_metric(self.val_data.y, val_pred, **self.pred_metric_kwargs)
            test_acc = self.pred_metric(self.test_data.y, test_pred, **self.pred_metric_kwargs)
            
        search_dict = dict({})
        search_dict["train_acc"] = train_acc
        search_dict["val_acc"] = val_acc
        search_dict["test_acc"] = test_acc
        search_dict["model"] = model
        search_dict["user_function"] = user_function
        return search_dict

    def search(self, clfs, clfs_space, hops_list=DEF_HOPS, user_functions=DEF_USER_FUNCTIONS,  attention_configs=DEF_ATTENTION_CONFIGS):
        store = dict({})
        for clf in tqdm(clfs):
            clf_name = clf().__class__.__name__
            space = clfs_space[clf_name]
            store[clf_name] = dict({})
            best_search_dict = None
            best_val = float("-inf")
            for hops in tqdm(hops_list):
                for attention_config in tqdm(attention_configs):
                    for user_function in user_functions:
                        search_dict = self.search_hop_clf_attention_config(hops, clf, user_function, attention_config, space)
                        if search_dict["val_acc"] >= best_val:
                            best_val = search_dict["val_acc"]
                            best_search_dict = search_dict
                            best_search_dict["attention_config"] = attention_config
                            best_search_dict["hops_list"] = hops
            store[clf_name] = best_search_dict
        return store