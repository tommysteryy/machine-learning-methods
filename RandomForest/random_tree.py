from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.lofTrees = []

    def fit(self, X, y):

        for i in range(self.num_trees):
            rt = RandomTree(self.max_depth)
            rt.fit(X,y)
            self.lofTrees.append(rt)

    def predict(self, X_hat):

        n = X_hat.shape[0]

        predictions = np.zeros((self.num_trees, n))

        # for i in range(self.num_trees):
        #     predictions[i] = self.lofTrees[i].predict(X_hat)

        for rt in self.lofTrees:
            predictions[self.lofTrees.index(rt)] = rt.predict(X_hat)

        all_preds = np.transpose(predictions)

        final_preds = np.zeros(n)
        for i in range(n):
            final_preds[i] = utils.mode(all_preds[i])

        return final_preds

        # for i in range(self.num_trees):
        #     predictions[i] = self.lofTrees[i].predict(X_hat)

        # final_preds = np.zeros((n, self.num_trees))
        # for example_index in range(n):

        #     ranForst_preds = np.zeros(self.num_trees)

        #     for rt_index in range(self.num_trees):
        #         ranForst_preds[rt_index] = predictions[rt_index][example_index]




        # return utils.mode(predictions)
            


