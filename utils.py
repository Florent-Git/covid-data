import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition as skd

def plot_pca_features(pca: skd.PCA, x: pd.DataFrame, features=None):
    if features is None:
        features = x.columns
        
    features_indexes = [x.columns.get_loc(f) for f in features]
        
    plt.xlabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0]*100, 2)))
    plt.ylabel('PC2 ({}%)'.format(round(pca.explained_variance_ratio_[1]*100, 2)))
    plt.grid()

    coef = pca.components_

    for feature in features_indexes:
        plt.arrow(0, 0, coef[0, feature], coef[1, feature], color='r', alpha=0.5)
        plt.text(coef[0, feature]*1.15, coef[1, feature]*1.15, x.columns[feature], color='g', ha='center', va='center', fontsize=12)