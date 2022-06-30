from lightgbm import LGBMRegressor, LGBMClassifier, LGBMRanker
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx
from .utils import *
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

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


def draw_net(G, gene=False, layout='2', font_size=8):
    b = nx.betweenness_centrality(G)
    labels = {}
    for node in G.nodes():
        if node:
            if node in gene:
                labels[node] = node
        else:
            if G.degree[node] >= 1 or b[node] >= 0.08:
                labels[node] = node
    layouts = {'1': nx.spring_layout(G),
               '2': nx.kamada_kawai_layout(G),
               '3': nx.shell_layout(G)}

    d = dict(G.degree)
    pos = layouts[layout]
    nx.draw(G,
            pos=pos,
            with_labels=False,
            nodelist=d.keys(),
            node_size=1500, alpha=0.7, node_color='lightblue', style='dashed', width=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_color='black')
    plt.margins(x=0.4, y=0.4)
    plt.show()


class EvoManager:
    def __init__(self, *args):
        self.samples = dict(zip(list(range(len(args))), args))
        self.predictions = dict()
        self.run_predictions()
        self.network = self.get_merged_network()
        self.cell_types_similarity = self.generate_comparison_df()

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
        comment_ = 1
        for s1 in self.samples:
            for s2 in self.samples:
                # if organism1!=organism2:
                run_id = [s1, s2]
                logging.info(f'Running {str(comment_)} out of 4')
                comment_ += 1
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

    def generate_comparison_df(self):
        return pd.concat([pd.concat([self.predictions['0_0'].apply(zscore),
                                     self.predictions['0_1'].apply(zscore)]),
                          pd.concat([self.predictions['1_0'].apply(zscore),
                                     self.predictions['1_1'].apply(zscore)])], axis=1)

    def cluster_map(self, visible_legend=False, c_map="viridis"):
        res = sns.clustermap(self.cell_types_similarity.T.corr(), cmap=c_map, yticklabels=True, figsize=(10, 10))
        res.cax.set_visible(visible_legend)
        plt.show()

    def get_closest_cell_types(self, cell_type, type_='matrix'):
        """
        Returns sorted list of closest celltypes
        """
        assert type_ in ['model', 'matrix']
        if type_ == 'model':
            return list(self.cell_types_similarity.sort_values(cell_type, ascending=False).index)
        else:
            return list(self.cell_types_similarity.T.corr().sort_values(cell_type, ascending=False).index)

    def check_(self, closest_cell_types, short_path):
        for i in [x for x in short_path if x in list(self.network['Cell_type'].unique())]:
            if i not in list(set(closest_cell_types) & set(list(self.network['Cell_type'].unique()))):
                return False
        return True

    @staticmethod
    def write_connections(in_, out_, raw_lists_for_debug, short_path_):
        raw_lists_for_debug.append(short_path_)
        for j, k in enumerate(short_path_):
            try:
                out_.append(short_path_[j + 1])
                in_.append(short_path_[j])
            except IndexError:
                continue

    def generate_cell_type_network(self, cluster1, cluster2, number_of_short_paths=100,
                                   minimal_number_of_nodes=3, get_only_direct=False, closest_clusters=10, net=True):
        in_, out_, raw_lists_for_debug = [], [], []
        subnet = pd.DataFrame()
        for short_path_ in [
            x for x in self.get_shortest_paths(self.network,
                                               cluster1, cluster2, number_of_short_paths)
        ]:
            if get_only_direct:
                if len(short_path_) == minimal_number_of_nodes and self.check_(
                        self.get_closest_cell_types(cluster2)[:closest_clusters], short_path_):
                    self.write_connections(in_, out_, raw_lists_for_debug, short_path_)
            elif len(short_path_) >= minimal_number_of_nodes and self.check_(
                    self.get_closest_cell_types(cluster2)[:closest_clusters], short_path_):
                self.write_connections(in_, out_, raw_lists_for_debug, short_path_)
        subnet['in'], subnet['out'] = in_, out_
        graph = nx.from_pandas_edgelist(subnet, 'in', 'out')
        if net:
            return graph
        else:
            return raw_lists_for_debug
