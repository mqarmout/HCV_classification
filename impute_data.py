import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def impute(dataset_path: str, targets: str):
    data = pd.read_csv(filepath_or_buffer="hcvdat0.csv", index_col=[0])

    y = data[targets]
    X = data.drop([targets], axis=1)

    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

    data_split_catagories = pd.get_dummies(X[categorical_cols])
    X = pd.concat([X, data_split_catagories], axis=1)
    X = X.drop("Sex", axis=1)

    numerical_transformer = Pipeline(steps = [
        ('imputer', IterativeImputer(max_iter=10, random_state=0)),    
        ('scale', StandardScaler())
    ])
    imputed_data = pd.DataFrame(numerical_transformer.fit_transform(X), columns=X.columns)
    final_data = pd.concat([imputed_data, y], axis=1)
    final_data.dropna(axis=0, inplace=True)
    return final_data