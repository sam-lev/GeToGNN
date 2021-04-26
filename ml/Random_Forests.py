from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

class RandomForest:

    def __init__(self):
        self.details = "Random forest classifier with feature importance"

    def random_forest_classifier(self, features=None, labels=None, train_features=None, train_labels=None,
                      test_features=None, test_labels=None, feature_map=False,
                      n_trees=10, depth=4, index_list=None):
        print("_____________________________________________________")
        print("                Random_Forest     ")
        print("number trees: ", n_trees)
        print("depth: ", depth)
        # Import the model we are using
        # Instantiate model with 1000 decision trees
        if feature_map:
            feature_map = self.arc_feature_map
            train_attributes = feature_map['train']
            train_features = np.array(train_attributes[1])
            train_labels = np.array(train_attributes[0])

            test_attributes = feature_map['test']
            test_features = np.array(test_attributes[1])
            test_labels = np.array(test_attributes[0])
        else:
            train_features = np.array(train_features)
            train_labels = np.array(train_labels)
            if test_features is not None:
                test_features = np.array(test_features)
                test_labels = np.array(test_labels)

        wp = float(np.sum(train_labels)) / float(len(train_labels))
        # wp = 1./wp

        pos_count = 2. * np.sum(train_labels)
        neg_count = 2. * (len(train_labels) - pos_count)
        # wp = len(train_labels) / pos_count
        # wn = len(train_labels) / neg_count
        num_n = 1 - train_labels
        wn = float(np.sum(num_n)) / float(len(train_labels))

        # wn = float(len(train_labels)-np.sum(train_labels) - 1) / float(len(train_labels) - 1)
        # bp = float(len(train_labels)) / 2*np.sum(train_labels)
        # nl = 1 - train_labels
        # bn = float(len(train_labels)) / 2*np.sum(nl)
        print("Using class weights for negative: ", wn)
        print("Using class weights for positive: ", wp)

        rf = RandomForestClassifier(max_depth=depth,
                                    n_estimators=n_trees)  # ,random_state=66, class_weight={0:wn,1:wp})#,random_state=666)#, class_weight={0:wp,1:1.})
        # n_estimators=1000, random_state=42)  # Train the model on training data

        rf.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        if test_features is not None:
            # self.preds = rf.predict(test_features)  # Calculate the absolute errors

            # self.preds = [p[1] for p in self.preds]

            # test_features = np.array([arc.features for arc in self.msc.arcs])
            pred_proba_test = rf.predict_proba(test_features)

            # pred_proba_train = rf.predict_proba(train_features)
            # train_arcs = [arc for arc in self.msc.arcs if arc.partition == 'train']
            # test_arcs = [arc for arc in self.msc.arcs if arc.partition == 'test']
            # for arc, pred in zip(train_arcs, list(pred_proba_train)):
            #    arc.prediction = pred[1]#[1-pred[0], pred[0]]
            print("Forest pred sample ", pred_proba_test[0])
            for arc, pred in zip(self.msc.arcs, list(pred_proba_test)):
                arc.prediction = pred

            # arcs = train_arcs + test_arcs
            # self.msc.arcs = self.arcs

            # self.pred_proba_train = []
            # train_labels = []

            # pred_proba = list(pred_proba_test) + list(pred_proba_train)

            # labels = np.array([int(arc.label[1]) for arc in self.msc.arcs])

            # list(test_labels) + list(train_labels)
            preds = pred_proba_test  # np.array([[1-p[1],p[1]] for p in pred_proba_test])#
            preds[preds >= 0.5] = 1.
            preds[preds < 0.5] = 0.
            preds = [l[len(l) - 1] for l in preds]

            # if len(test_labels[0]) == 2:
            #    test_labels = [l[1] for l in test_labels]

            # errors = abs(self.preds - self.labels)  # Print out the mean absolute error (mae)
            # round(np.mean(errors), 2), 'degrees.')
            print('----------------------------')
            mse = rf.score(test_features, test_labels)  # np.array(list(test_features) + list(train_features)),
            #                                       np.array(list(test_labels) + list(train_labels)))
            print('Mean Absolute Error:', mse)
            p, r, fs = self.compute_quality_metrics(preds, test_labels, for_print=mse)
            precision, recall = self.precision_and_recall(predictions=preds, labels=test_labels)

            return [mse, precision, recall], p, r, fs

    def feature_importance(self, features, labels, filename=None,  n_informative = 3, plot=False):
        # Build a classification task using 3 informative features
        if self.number_features is None and self.features is not None:
            self.number_features = self.features[0].shape[0]

        #X, y = make_classification(n_samples=1000,
        #                           n_features=self.number_features,
        #                           n_informative=n_informative,
        #                           n_redundant=0,
        #                           n_repeated=0,
        #                           n_classes=2,
        #                           random_state=0,
        #                           shuffle=False)

        # Build a forest and compute the impurity-based feature importances
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=666)

        X = np.array(features)
        y = np.array(labels)
        forest.fit(X, y)
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        if self.feature_names is not None and filename is not None:
            feat_importance_file = open(filename+"featImportance.txt", "w+")
            for f in range(len(self.feature_names)):#X.shape[1]):
                feat_importance_file.write(str(f + 1) + ' ' + self.feature_names[indices[f]] + ' ' + str(indices[f])
                                           +' '+str(importances[indices[f]])+"\n")
            feat_importance_file.close()

        # Plot the impurity-based feature importances of the forest
        if plot:
            plt.figure()
            plt.title("Feature importances")
            plt.bar(range(X.shape[1]), importances[indices],
                    color="r", yerr=std[indices], align="center")
            plt.xticks(range(X.shape[1]), indices)
            plt.xlim([-1, X.shape[1]])
            plt.show()
            return indices, self.feature_names