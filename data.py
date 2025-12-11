import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScalar

df = pd.read_csv("file goes here")

X = df.drop('Class', axis = 1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_tranform(X_train)
X_test = scaler.transform(X_test)
