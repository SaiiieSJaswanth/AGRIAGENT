from sklearn.model_selection import train_test_split

def split_data(X, y, task_kind, random_state=42, test_size=0.2):
    stratify = y if task_kind == "classification" else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
