#from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import classification_report
from sklearn.svm import SVC
#import utils.tools as utils
import numpy as np
#from sklearn.model_selection import StratifiedKFold

dataSet = np.loadtxt('C:\\Users\\admin\\Desktop\\test.txt', delimiter=' ')
X = dataSet[:,0:21]
Y = dataSet[:,21]

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
                     'C': [10, 100, 1000, 2000, 3000]}]
                    
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
clf = GridSearchCV(SVC(), tuned_parameters, cv=10,
                       scoring='%s_weighted' % score)

clf.fit(X, Y)


print("Best parameters set found on development set:")
print()

print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()































