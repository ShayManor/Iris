from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt



def get_acc(size: float):
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    model = LogisticRegression(multi_class='multinomial', max_iter=100000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

x = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
y = []
for val in x:
    y.append(get_acc(val))
plt.plot(x, y)
plt.xlabel('% dataset used for testing')
plt.ylabel('Accuracy')
plt.show()
