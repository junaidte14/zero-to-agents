from sklearn.neighbors import KNeighborsClassifier

X_train = [[0.1, 0.1], [0.9, 0.9], [0.2, -0.1], [1.2, 0.3], [-0.3, 0.2]]
y_train = [1, 0, 1, 0, 1]

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

print(clf.predict([[0.4, 0.4]]))  # generalizes, same principle, production-grade implementation