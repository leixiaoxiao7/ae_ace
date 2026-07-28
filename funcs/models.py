from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

from dataCook import *
from tools import *


def accessLR(X_train, y_train, X_test, y_test, outName=None):
    # 创建带类别权重的模型
    model = LogisticRegression(class_weight='balanced', solver='liblinear')
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    print("Accuracy:", model.score(X_test, y_test))
    print("Consusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    #print("模型系数:", model.coef_)
    if outName is not None:
        savePklMode(outName+'_lr', model)


def accessRF(X_train, y_train, X_test, y_test, class_weight=None, treeNum=100, outName=None):

    clf = RandomForestClassifier(
        n_estimators=treeNum, max_depth=10, random_state=42, class_weight=class_weight)
    clf.fit(X_train, y_train)

    y_pred = list(clf.predict(X_test))

    print("Accuracy:", clf.score(X_test, y_test))
    print("Consusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    if outName is not None:
        savePklMode(outName+'_rf', clf)


def accessSVC(X_train, y_train, X_test, y_test, outName=None):

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    y_pred = list(clf.predict(X_test))

    print("Accuracy:", clf.score(X_test, y_test))
    print("Consusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    if outName is not None:
        savePklMode(outName+'_svc', model)


def pcaAnalysis(X, compNum=10):
    # Step 1: Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    # Step 2: Apply PCA
    # Select the number of principal components
    pca = PCA(n_components=compNum)
    principal_components = pca.fit_transform(scaled_data)

    # Step 3: Create a new dataframe
    newFea = []
    for k in range(compNum):
        newFea.append('pc'+str(k+1))

    principal_df = pd.DataFrame(data=principal_components, columns=newFea)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    print("Principal Components:\n", principal_df)
    print("Explained variance ratio:\n", explained_variance_ratio)

    return principal_df
