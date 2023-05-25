import pandas as pd
from dataAnalysis.DataAnalysis import DataAnalysis
from dataAnalysis.Subplot import Subplot
from dataAnalysis.neo4j.Upload import Upload
from hgnn.HGNN import HGNN
from hgnn.HGraph import HGraph

subplot = Subplot(1, 2)
data = pd.read_csv(r"extdata/sbcdata.csv", header=0)
# data = pd.read_csv(r"extdata/sbc_small.csv", header=0)
data_analysis = DataAnalysis(data)

# upload = Upload(data_analysis.get_training_data())
# upload.upload_patients()
# upload.upload_feature_nodes()
#
train_graph = HGraph(data_analysis.get_training_data())
train_graph.add_features()

test_graph = HGraph(data_analysis.get_testing_data())
test_graph.add_features()

hgnn = HGNN(train_graph.get_graph(), test_graph.get_graph())
# data_analysis.build_graph()

# data_analysis.logistic_regression()
# data_analysis.extra_trees()
# data_analysis.random_forest()
# data_analysis.rus_boost()
# data_analysis.dialnd_rus_boost()
# data_analysis.bagging_classifier()
# data_analysis.decision_tree()
# data_analysis.k_neighbors()

# data_analysis.neural_network()
# data_analysis.xg_boost() ##change y to int labels

# data_analysis.logistic_regression_grid_search()
# data_analysis.logistic_regression_rus_grid_search()

# data_analysis.rus_boost_grid_search()
# data_analysis.grid_search()

# data_analysis.tpot()
# data_analysis.logistic_regression_ros_grid_search()
# data_analysis.logistic_regression_smote_grid_search()
# data_analysis.logistic_regression_svm_smote_grid_search()
# data_analysis.logistic_regression_adasyn_grid_search()

# data_analysis.mlp_grid_search()
# data_analysis.xg_boost()
# data_analysis.logistic_regression_grid_search()

# data_analysis.lazy_predict()
# data_analysis.support_vector_machine()
# data_analysis.show_diagrams()
# # data_analysis.show_text_information()
# data_analysis.show_comparison_diagrams()