import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import warnings
import joblib
warnings.filterwarnings("ignore")
df=pd.read_csv("Dataset1.csv")
print(df)
X = df.iloc[: , 0:-1]
Y = df.iloc[: , -1]
print(Y.value_counts())
X,Y=SMOTE().fit_resample(X,Y)
print(X.shape,Y.shape)
print(Y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
print('Training dataset shape:', X_train.shape, Y_train.shape)
print('Testing dataset shape:', X_test.shape, Y_test.shape)

Y_train_resample_flat = Y_train.to_numpy().ravel()
Y_test_resample_flat = Y_test.to_numpy().ravel()

print('Training dataset shape:', X_train.shape, Y_train_resample_flat.shape)
print('Testing dataset shape:', X_test.shape, Y_test_resample_flat.shape)
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
forward_fs = sfs(rf , k_features=8,forward=True,floating=False,verbose=2,scoring='accuracy',cv=5)
forward_fs = forward_fs.fit(X_train, Y_train_resample_flat)
feat_names = list(forward_fs.k_feature_names_)
print(feat_names)
X_train_new=X_train[['D.O. (mg/l)', 'Temp', 'PH', 'Turbidity (NTU)', 'CONDUCTIVITY (µmhos/cm)', 'B.O.D. (mg/l)', 'NITRATENAN N+ NITRITENANN (mg/l)', 'TOTAL COLIFORM (MPN/100ml)Mean']]
X_test_new=X_test[['D.O. (mg/l)', 'Temp', 'PH', 'Turbidity (NTU)', 'CONDUCTIVITY (µmhos/cm)', 'B.O.D. (mg/l)', 'NITRATENAN N+ NITRITENANN (mg/l)', 'TOTAL COLIFORM (MPN/100ml)Mean']]
rf_model=rf.fit(X_train_new,Y_train_resample_flat)
# lr=LogisticRegression(random_state=0,max_iter=10)
# lr_model=lr.fit(X_train_new,Y_train_resample_flat)
# gnb=GaussianNB()
# gnb_model=gnb.fit(X_train_new,Y_train_resample_flat)
# knn=KNeighborsClassifier()
# knn_model=knn.fit(X_train_new,Y_train_resample_flat)
def print_Score(clf,x_train,x_test,y_train,y_test,train=True):
    if train:
        pred=clf.predict(x_train)
        clf_report=pd.DataFrame(classification_report(y_train,pred,output_dict=True))
        print("Train Result:")
        print(f"Accuracy Score:{accuracy_score(y_train,pred)*100:.2f}%")
        print("---------------------------------")
        print(f"Confusion Matrix:\n{confusion_matrix(y_train,pred)}\n")
    elif train==False:
        pred=clf.predict(x_test)
        clf_report=pd.DataFrame(classification_report(y_test,pred,output_dict=True))
        print("Test Result:")
        print(f"Accuracy Score:{accuracy_score(y_test,pred)*100:.2f}%")
        print("---------------------------------")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test,pred)}\n")
print_Score(rf_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=True)
print_Score(rf_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=False)
# lr_train_score=print_Score(lr_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=True)
# lr_test_score=print_Score(lr_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=False)
# gnb_train_score=print_Score(gnb_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=True)
# gnb_test_score=print_Score(gnb_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=False)
# print("this is the guassian accuracy")
# knn_train_score=print_Score(knn_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=True)
# knn_test_score=print_Score(knn_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=False)
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score, KFold
# import pandas as pd

# clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# scores = cross_val_score(clf, X_train_new,Y_train_resample_flat, cv=kf)

# clf.fit(X_train_new,Y_train_resample_flat)
# dtc=DecisionTreeClassifier(random_state=0,max_depth=10,min_samples_split=5)
# dt_model=dtc.fit(X_train_new,Y_train_resample_flat)

# dt_train_score=print_Score(dt_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=True)
# dt_test_model=print_Score(dt_model,X_train_new,X_test_new,Y_train_resample_flat,Y_test_resample_flat,train=False)

# import joblib
# joblib.dump(rf,'final.pkl')
# print(X_train_new.head())
# m=joblib.load('final.pkl')
# x=m.predict(X_train_new)
# print(list(x).index(1))
# print(X_train_new.iloc[4])
# lr=LogisticRegression(random_state=0,max_iter=10)





