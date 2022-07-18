import torch

class PCA():
    def __init__(self, threshold=0.9):
        assert 0.0 <= threshold <= 1.0
        self.threshold = threshold
        
    def fit(self, X):
        """
        X - (n_samples, n_features)
        Reference: https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/decomposition/_pca.py
        """
        assert X.ndim == 2
        n_samples, n_features = X.size()[0], X.size()[1]
        
        self.mean_ = X.mean(dim=0)
        X = X - self.mean_
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        components_ = Vt
        
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        
        ratio_cumsum = torch.cumsum(explained_variance_ratio_, dim=0)
        n_components = torch.searchsorted(ratio_cumsum, self.threshold, right=True) + 1
        
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        
    def transform(self, X):
        """
        X - (n_samples, n_features)
        Reference: https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/decomposition/_base.py
        """
        X = X - self.mean_
        return X.mm(self.components_.t())