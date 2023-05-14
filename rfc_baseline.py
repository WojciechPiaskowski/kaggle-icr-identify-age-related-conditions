# imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

pd.set_option('display.max_columns', 200)


# balanced logloss function
def calculate_balanced_logloss(y, y_pred):
    n0 = np.sum(1 - y)
    n1 = np.sum(y)
    n = y.count()

    w_0 = n / n0
    w_1 = n / n1

    idx_0 = np.where(y == 0)
    idx_1 = np.where(y == 1)

    balanced_logloss = ((-w_0 / n0) * np.sum(np.log(y_pred[idx_0, 0])) - (w_1 / n1) * np.sum(np.log(y_pred[idx_1, 1]))) / (w_0 + w_1)

    return balanced_logloss


# balanced log loss metric
metric_balanced_log_loss = make_scorer(calculate_balanced_logloss, greater_is_better=True)


# basic load/transform function
def etl(path):
    df = pd.read_csv(path)
    print('df shape: ', df.shape)

    # one hot encode categorical column 'EJ'
    temp = pd.get_dummies(df['EJ'])
    temp_cols = ['EJ_' + str(temp_col) for temp_col in temp.columns]
    temp.columns = temp_cols
    try:
        temp = temp.drop('EJ_B', axis=1)
    except KeyError:
        pass
    df = pd.concat([df.drop('EJ', axis=1), temp], axis=1)
    del temp

    # simple mean imputation
    for col_name in df.columns:
        if col_name not in ['Id', 'Class']:
            df[col_name] = df[col_name].apply(lambda x: df[col_name].mean() if pd.isna(x) else x)

    return df


# load/transform training data
df_train = etl('train.csv')
# target column
target = 'Class'

# list train columns
train_cols = []
remove_cols = ['Id', 'Class']
for col in df_train.columns:
    if col not in remove_cols:
        train_cols.append(col)

# split into train and test(valid) sets
x_train, x_test, y_train, y_test = train_test_split(df_train[train_cols], df_train[target],
                                                    test_size=0.15, random_state=23)

# fit baseline random forest model
model = RandomForestClassifier(max_depth=5, random_state=23)
model.fit(x_train, y_train)
yhat = model.predict_proba(x_test)

# calculate balanced log loss
b_logloss = calculate_balanced_logloss(y_test, yhat)
print('balanced log loss: ', b_logloss)

# load test data, make predictions and save results to submission.csv
df_test = etl('test.csv')
yhat_test = pd.DataFrame(model.predict_proba(df_test[train_cols]))
yhat_test.columns = ['class_0', 'class_1']
submission = pd.concat([df_test['Id'], yhat_test], axis=1)
submission.to_csv('submission.csv', index=False)
