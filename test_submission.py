import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
pd.set_option('display.max_columns', 200)

def balanced_log_loss(y_true, y_pred):
  # calculate the number of observations for each class
  N_0 = np.sum(1 - y_true)
  N_1 = np.sum(y_true)
   # calculate the weights for each class
  w_0 = 1 / N_0
  w_1 = 1 / N_1
   # calculate the predicted probabilities for each class
  p_0 = np.clip(y_pred[:, 0], 1e-15, 1 - 1e-15)
  p_1 = np.clip(y_pred[:, 1], 1e-15, 1 - 1e-15)
   # calculate the log loss for each class
  log_loss_0 = -w_0 * np.sum(y_true * np.log(p_0))
  log_loss_1 = -w_1 * np.sum(y_true * np.log(p_1))
   # calculate the balanced logarithmic loss
  balanced_log_loss = (log_loss_0 + log_loss_1) / (w_0 + w_1)
    return balanced_log_loss

metric_balanced_log_loss = make_scorer(balanced_log_loss, greater_is_better=True)

df_train = pd.read_csv('train.csv')
df_train.shape
df_train.head()

target = 'Class'
train_cols = []
remove_cols = ['Id', 'Class']
for col in df_train.columns:
    if col not in remove_cols:
        train_cols.append(col)

df_train['EJ_isB'] = pd.get_dummies(df_train['EJ'], drop_first=True)
df_train = df_train.drop('EJ', axis=1)
train_cols = [element if element != 'EJ' else 'EJ_isB' for element in train_cols]

# simple mean imputation
for col in train_cols:
    df_train[col] = df_train[col].apply(lambda x: df_train[col].mean() if pd.isna(x) else x)


rfc = RandomForestClassifier()
rfc.fit(df_train[train_cols], df_train[target])
yhat = rfc.predict_proba(df_train[train_cols])

cv = cross_val_score(rfc, df_train[train_cols], df_train[target], cv=10, scoring='f1')



def balanced_log_loss(y_true, y_pred):
  # calculate the number of observations for each class
  N_0 = np.sum(1 - y_true)
  N_1 = np.sum(y_true)
   # calculate the weights for each class
  w_0 = 1 / N_0
  w_1 = 1 / N_1
   # calculate the predicted probabilities for each class
  p_0 = np.clip(y_pred[:, 0], 1e-15, 1 - 1e-15)
  p_1 = np.clip(y_pred[:, 1], 1e-15, 1 - 1e-15)
   # calculate the log loss for each class
  log_loss_0 = -w_0 * np.sum(y_true * np.log(p_0))
  log_loss_1 = -w_1 * np.sum(y_true * np.log(p_1))
   # calculate the balanced logarithmic loss
  balanced_log_loss = (log_loss_0 + log_loss_1) / (w_0 + w_1)
    return balanced_log_loss


score = balanced_log_loss(df_train[target], yhat)
print(score)






y = df_train[target]

N_0 = np.sum(1 - df_train[target])
N_1 = np.sum(df_train[target])
N = df_train[target].count()
# calculate the weights for each class
# w_0 = 1 / N_0
# w_1 = 1 / N_1

w_0 = N / N_0
w_1 = N / N_1

y






