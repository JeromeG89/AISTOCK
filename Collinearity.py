from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def remove_HighCorrelation(X, threshold):
    correlation_matrix = X.corr()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    correlation_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()

    high_corr_pairs = np.where((correlation_matrix.abs() > threshold) & (correlation_matrix != 1))
    high_corr_features = set()

    for i, j in zip(*high_corr_pairs):
        if i < j:  # Avoid duplicates
            high_corr_features.add(correlation_matrix.columns[i])
            high_corr_features.add(correlation_matrix.columns[j])
    high_corr_features = list(high_corr_features)
    if high_corr_features:
        pca = PCA(n_components=min(len(high_corr_features), X.shape[0]))
        data_high_corr = X[high_corr_features]
        data_high_corr_scaled = scaler.fit_transform(data_high_corr)
        pca_result = pca.fit_transform(data_high_corr_scaled)
        
        # Create a DataFrame for PCA results
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
        X_PCA = X.drop(columns=high_corr_features).reset_index().join(pca_df).set_index('Date')
        return X_PCA