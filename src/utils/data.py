import numpy as np


from sklearn.datasets import make_classification



class data():

    @classmethod
    def dataRandomGenerator(self,n_samples,n_features,n_classes):

        X, Y = make_classification(n_samples=n_samples, n_features=n_features,n_redundant=0, n_clusters_per_class=1, n_classes=n_classes)

        return X,Y