print('-KNEAREST NEIGHBOURS FTW 🎊')
print('handling imports...')
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

print('loading & proccessing dataset...')
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
print('training...')

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"trained with {accuracy} accuracy")

example_measures = np.array([[8,3,1,1,1,2,3,2,1]])
prediction = clf.predict(example_measures)
print(f"Result of example measures prediction: " + ("safe" if prediction[0]==2 else "not safe 😞"))