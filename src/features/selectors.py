from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression

def select_k_features(X_train, y_train, task_kind, k=60):
    score_func = mutual_info_classif if task_kind == "classification" else mutual_info_regression
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X_train, y_train)
    return selector
