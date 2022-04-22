from lightgbm import LGBMRegressor, LGBMClassifier, LGBMRanker
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx
from .utils import *

logging.basicConfig(level=logging.INFO)


class Sample:
    def __init__(self, matrix, cell_types):
        self.models = {}
        self.matrix = matrix
        self.cell_types = list(cell_types)
        self.cell_types_df = pd.DataFrame()
        self.models_top_features = {}

        self.generate_ys()
        self.get_all_models()

    def generate_ys(self):
        """
        Modifies list of N cell states into N lists of 1/0 state for each cell
        """
        for cell_type1 in list(set(self.cell_types)):
            list_ = []
            for cell_type2 in self.cell_types:
                if cell_type2 == cell_type1:
                    list_.append(1)
                else:
                    list_.append(0)
            self.cell_types_df[cell_type1] = list_

    def generate_model(self, cluster_col):
        """
        Generates LGBM model for a given Y
        """
        X_train, X_test, y_train, y_test = train_test_split(self.matrix,
                                                            cluster_col,
                                                            test_size=0.25,
                                                            random_state=42)
        model = LGBMRegressor(n_estimators=500, first_metric_only=True)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric=['auc'],
                  early_stopping_rounds=10,
                  verbose=0)
        top_features = pd.DataFrame(
            [X_train.columns,
             model.feature_importances_]).T.sort_values(1, ascending=False)
        X_train_upd = sigmoid(update_df(X_train, list(top_features[0][:3000])))
        X_test_upd = sigmoid(update_df(X_test, list(top_features[0][:3000])))
        model_upd = LGBMRegressor(n_estimators=500, first_metric_only=True)
        model_upd.fit(X_train_upd, y_train,
                      eval_set=[(X_test_upd, y_test)],
                      eval_metric=['auc'],
                      early_stopping_rounds=10,
                      verbose=0)
        top_features_ = pd.DataFrame(
            [X_train_upd.columns,
             model_upd.feature_importances_]).T.sort_values(1, ascending=False)
        return {
            'model': model_upd,
            'features': top_features_,
        }

    def get_all_models(self):
        """
        Writes generated models into Sample attributes
        """
        for cell_type_ in self.cell_types_df.columns:
            model = self.generate_model(self.cell_types_df[cell_type_])
            self.models[cell_type_] = model['model']
            self.models_top_features[cell_type_] = model['features']
            logging.info(f'Model for {cell_type_} is generated')


class EvoManager:
    def __init__(self, *args):
        self.samples = dict(zip(list(range(len(args))), args))
        self.predictions = dict()
        self.run_predictions()
        self.network = self.get_merged_network()

    @staticmethod
    def prepare_input_df(matrix, top_features, c=0):
        """
        Replaces missing genes with NaN values
        """
        df = pd.DataFrame()
        for i in top_features:
            if i in matrix.columns:
                df[i] = list(matrix[i])
            else:
                # c += 1
                df[i] = [float('NaN')] * matrix.shape[0]
        # logging.info(f'C score is {c}')
        return sigmoid(df)

    @staticmethod
    def get_means(df):
        """
        Returns mean score for each cell state
        """
        df_ = pd.DataFrame()
        for i in [x for x in df.columns if 'cluster' not in x]:
            df_[i] = list(df.groupby('clusters', as_index=False)[i].mean()[i])
        df_.index = list(
            df.groupby('clusters', as_index=False)[i].mean()['clusters'])
        return df_

    def between_two(self, sample1, sample2):
        """
        Runs all the models of the sample 1 on the sample2
        """
        df = pd.DataFrame()
        df['clusters'] = sample2.cell_types
        for cell_type_model in sample1.models:
            top_features_ = list(
                sample1.models_top_features[cell_type_model][0])
            res = sample1.models[cell_type_model].predict(
                self.prepare_input_df(sample2.matrix, top_features_))
            df[cell_type_model] = res
        return self.get_means(df)

    def run_predictions(self):
        """
        Predicts cell states for each input sample
        """
        for s1 in self.samples:
            for s2 in self.samples:
                # if organism1!=organism2:
                run_id = [s1, s2]
                print(run_id)
                self.predictions['_'.join([str(x) for x in run_id
                                           ])] = self.between_two(
                    self.samples[s1],
                    self.samples[s2])

    @staticmethod
    def extract_features(dct, krt_remove=True):
        df = None
        for cell_type_ in dct:
            df_ = pd.DataFrame()
            df_['Cell_type'] = [cell_type_] * 75
            df_['Gene'] = list(dct[cell_type_][0])[:75]
            df_['Importance'] = list(dct[cell_type_][1])[:75]
            if krt_remove:
                df_ = df_[~df_.Gene.str.contains("KRT")]
            if df is None:
                df = df_
            else:
                df = pd.concat([df, df_])
        return df

    def get_merged_network(self):
        """
        Generates cell_state/gene_program network
        """
        return pd.concat([
            self.extract_features(self.samples[x].models_top_features)
            for x in self.samples
        ])

    @staticmethod
    def get_shortest_paths(df, in_, out_, k=10):
        network = nx.from_pandas_edgelist(df, 'Gene', 'Cell_type', ['Importance'])
        x = nx.shortest_simple_paths(network, in_, out_)
        for counter, path in enumerate(x):
            yield path
            if counter == k - 1:
                break


