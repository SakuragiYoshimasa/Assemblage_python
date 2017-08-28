'''''''''''''''
SVM Pipeline
'''''''''''''''
'''
pipe_lr = Pipeline([('sc', StandardScaler()), ('pca', PCA()), ('svc', SVC())])
parameter =  {'pca__n_components': [100, 140, 180, 220], 'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['linear', 'rbf', 'sigmoid'], 'svc__gamma':[0.0001, 0.001, 0.1]}
clf = GridSearchCV(pipe_lr, parameter, cv=5, n_jobs = -1)
score_average = 0
for n in range(Y_dev.shape[1]):
    clf.fit(X_train, Y_train[:, n])
    #print(clf.best_estimator_)
    print(n)
    print(clf.best_score_)
    print(clf.best_params_)
    score_average = score_average + clf.best_score_ / float(Y_dev.shape[1])
print('score average = %.3f' % score_average)
'''
'''
0
0.905
{'pca__n_components': 180, 'svc__C': 1000, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
1
0.94
{'pca__n_components': 100, 'svc__C': 1000, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
2
0.9175
{'pca__n_components': 100, 'svc__C': 1, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
3
0.9325
{'pca__n_components': 100, 'svc__C': 10, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
4
0.9425
{'pca__n_components': 140, 'svc__C': 100, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
5
0.9375
{'pca__n_components': 140, 'svc__C': 10, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
6
0.955
{'pca__n_components': 220, 'svc__C': 10, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
7
0.955
{'pca__n_components': 220, 'svc__C': 10, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
8
0.965
{'pca__n_components': 140, 'svc__C': 10, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
'''


'''''''''''''''
Random Forest Pipeline
'''''''''''''''
'''
pipe_lr = Pipeline([('sc', StandardScaler()), ('pca', PCA()), ('rf', RandomForestClassifier())])
parameters = {
    'pca__n_components': [100, 140, 180, 220],
    'rf__n_estimators': [20, 100, 300],
    'rf__max_features': [5, 10, 15],
    'rf__min_samples_split':[3, 5, 10],
    'rf__max_depth':[5, 10, 15]
    }

clf = GridSearchCV(pipe_lr, parameters, cv=5, n_jobs = -1)
score_average = 0
for n in range(Y_dev.shape[1]):
    clf.fit(X_train, Y_train[:, n])
    #print(clf.best_estimator_)
    print(n)
    print(clf.best_score_)
    print(clf.best_params_)
    score_average = score_average + clf.best_score_ / float(Y_dev.shape[1])
'''
'''
全部0.92ぐらい
'''

'''''''''''''''
Random Forest
'''''''''''''''
'''
parameters = {
        'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
        'max_features'      : [3, 5, 10, 15, 20],
        'random_state'      : [0],
        'n_jobs'            : [1],
        'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
        }

clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs = -1)
score_average = 0

for n in range(Y_dev.shape[1]):

    clf.fit(X_train, Y_train[:, n])
    #print(clf.best_estimator_)
    print(n)
    print(clf.best_score_)
    print(clf.best_params_)
    score_average = score_average + clf.best_score_ / float(Y_dev.shape[1])

print('score average = %.3f' % score_average)
'''
'''
0
0.8975
{'max_depth': 15, 'max_features': 5, 'min_samples_split': 5, 'n_estimators': 10, 'n_jobs': 1, 'random_state': 0}
1
0.925
{'max_depth': 10, 'max_features': 15, 'min_samples_split': 5, 'n_estimators': 20, 'n_jobs': 1, 'random_state': 0}
2
0.8825
{'max_depth': 5, 'max_features': 10, 'min_samples_split': 3, 'n_estimators': 5, 'n_jobs': 1, 'random_state': 0}
3
0.92
{'max_depth': 10, 'max_features': 20, 'min_samples_split': 3, 'n_estimators': 300, 'n_jobs': 1, 'random_state': 0}
4
0.92
{'max_depth': 5, 'max_features': 20, 'min_samples_split': 3, 'n_estimators': 20, 'n_jobs': 1, 'random_state': 0}
5
0.895
{'max_depth': 10, 'max_features': 20, 'min_samples_split': 5, 'n_estimators': 300, 'n_jobs': 1, 'random_state': 0}
6
0.9175
{'max_depth': 10, 'max_features': 10, 'min_samples_split': 5, 'n_estimators': 20, 'n_jobs': 1, 'random_state': 0}
7
0.8875
{'max_depth': 10, 'max_features': 20, 'min_samples_split': 5, 'n_estimators': 20, 'n_jobs': 1, 'random_state': 0}
8
0.9275
{'max_depth': 10, 'max_features': 15, 'min_samples_split': 3, 'n_estimators': 20, 'n_jobs': 1, 'random_state': 0}
score average = 0.908
'''

















'''
classifiers = []
for j in range(Y_dev.shape[1]):
    y = Y_dev[:, j]
    classifier = LogisticRegression(penalty='l2', C=0.01)
    classifier.fit(X_dev_pca, y)
    classifiers.append(classifier)

Y_val_pred = np.zeros(Y_val.shape)
for j in range(Y_dev.shape[1]):
    classifier = classifiers[j]
    y = classifier.predict_proba(X_val_pca)[:, 1]
    Y_val_pred[:, j] = y

roc_auc_score(Y_val, Y_val_pred, average='macro')

classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=0.01))
classifier.fit(X_dev_pca, Y_dev)
Y_val_pred = classifier.predict_proba(X_val_pca)

steps = [('scaler', StandardScaler()),
         ('decomposer', PCA(10, random_state=0)),
         ('classifier', OneVsRestClassifier(LogisticRegression(penalty='l2')))]
pipeline = Pipeline(steps)

params = {'classifier__estimator__C': [0.01, 0.1, 1.0, 10., 100.]}
scorer = make_scorer(roc_auc_score, average='macro', needs_proba=True)

predictor = GridSearchCV(pipeline, params, cv=5, scoring=scorer)
predictor.fit(X_dev, Y_dev)
print(predictor.best_params_)

Y_val_pred = predictor.predict_proba(X_val)
print(roc_auc_score(Y_val, Y_val_pred, average='macro'))

params = {'classifier__estimator__C': [0.01, 0.1, 1.0, 10., 100.],
         'decomposer__n_components': [10, 20, 50]}

predictor = GridSearchCV(pipeline, params, cv=5, scoring=scorer)
predictor.fit(X_dev, Y_dev)

Y_val_pred = predictor.predict_proba(X_val)
roc_auc_score(Y_val, Y_val_pred, average='macro')


final_predictor = predictor.best_estimator_
final_predictor.fit(X_train, Y_train)

Y_test_pred = final_predictor.predict_proba(X_test)
np.savetxt('submission.dat', Y_test_pred, fmt='%.6f')

if __name__ == '__main__':
    main()
'''
