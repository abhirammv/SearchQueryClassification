import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    #dataset = pd.read_csv('{0}.csv'.format(file_name), dtype={'query_id': 'category', 'url_id': 'category','is_homepage': 'category', 
     #                     'sig3': 'float64', 'sig4': 'float64', 'sig5': 'float64', 'sig6': 'float64'})
    
    return dataset

def plus_1(x): 
    if x == 0:
        return 0.001
    else:
        return x

def l_weight(x):
    if x == 1:
        return np.log(10.00)
    elif x == 2:
        return 15.50
    elif x == 3:
        return 9.00
    elif x == 4:
        return 4.35
    elif x == 5:
        return 3.20
    elif x == 6:
        return 0.84
    elif x == 7:
        return 0.42
    else:
        return 0.2
        

test_data = read_file("test")

train_data = read_file("training")

train_data['sig3'] = train_data['sig3'].apply(plus_1)
train_data['sig4'] = train_data['sig4'].apply(plus_1)
train_data['sig5'] = train_data['sig5'].apply(plus_1)
train_data['sig6'] = train_data['sig6'].apply(plus_1)

train_data['sig3'] = train_data['sig3'].apply(np.log)
train_data['sig4'] = train_data['sig4'].apply(np.log)
train_data['sig5'] = train_data['sig5'].apply(np.log)
train_data['sig6'] = train_data['sig6'].apply(np.log)

train_weight_data = train_data
train_weight_data['query_length'] = train_weight_data['query_length'].apply(l_weight)

test_data['sig3'] = test_data['sig3'].apply(plus_1)
test_data['sig4'] = test_data['sig4'].apply(plus_1)
test_data['sig5'] = test_data['sig5'].apply(plus_1)
test_data['sig6'] = test_data['sig6'].apply(plus_1)

test_data['sig3'] = test_data['sig3'].apply(np.log)
test_data['sig4'] = test_data['sig4'].apply(np.log)
test_data['sig5'] = test_data['sig5'].apply(np.log)
test_data['sig6'] = test_data['sig6'].apply(np.log)



#print(train_data.dtypes)

#min_sig8 = train_data.groupby('query_id', as_index = False)['sig8'].max().apply(plus_1)
#Havent modified train_weight_data 

tr_minsig8 = train_data.join(train_data.groupby('query_id')['sig8'].min(), on='query_id', rsuffix='_min')

te_minsig8 = test_data.join(test_data.groupby('query_id')['sig8'].min(), on='query_id', rsuffix='_min')


tr_urlc = train_data.join(train_data.groupby('query_id')['url_id'].count(), on='query_id', rsuffix='_count')

trw_minsig8 = train_weight_data.join(train_weight_data.groupby('query_id')['sig8'].min(), on='query_id', rsuffix='_min')
trw_urlc = train_weight_data.join(train_weight_data.groupby('query_id')['url_id'].count(), on='query_id', rsuffix='_count')
#print(type(group))


#for col in ['query_id', 'url_id', 'is_homepage']:
 #   train_data[col] = train_data[col].astype('category')

#Handling only training data from now

X_s8 = tr_minsig8.iloc[:,:].values
X = X_s8[:,[0,1,2,3,4,5,6,7,8,9,10,13]]
y = X_s8[:,12]

#remove 11 
X_te8 = te_minsig8.iloc[:,:].values
X_te = X_te8[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]]


Xw_urlc = trw_urlc.iloc[:,:].values
Xw = Xw_urlc[:,[0,1,2,3,4,5,6,7,8,9,10,11,13]]
yw = Xw_urlc[:,12]

X_w8 = trw_minsig8.iloc[:,:].values
Xw_sig8 = X_w8[:,[0,1,2,3,4,5,6,7,8,9,10,11,13]]
yw_sig8 = X_w8[:,12]



#Building Classification models
from sklearn.cross_validation import cross_val_score
def perform_cv(model, predictors, output_var, cv_fold):
    cross_validation = cross_val_score(model, predictors, output_var, scoring="accuracy", cv=cv_fold)
    if type(model).__name__ != "RFE":
        print(type(model).__name__ + " --> " + str(cross_validation.mean()))
        return cross_validation.mean()



'''#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()

#Min sig 8 
perform_cv(logreg_model, X, y, 10)
#LogisticRegression --> 0.612148229801

#query weight 
perform_cv(logreg_model, Xw, yw, 10)
#LogisticRegression --> 0.616533050187

#query_weight and min sig 8
perform_cv(logreg_model, Xw_sig8, yw_sig8, 10)
#LogisticRegression --> 0.612148229801

#url count and query weight
perform_cv(logreg_model, X_wc, y_wc, 10)
#LogisticRegression --> 0.616533050187

#____________________________________________
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()

#Min sig 8 
perform_cv(nb_model, X, y, 10)
#GaussianNB --> 0.593121648385

#query weight 
perform_cv(nb_model, Xw, yw, 10)
#GaussianNB --> 0.591572491664

#query_weight and min sig 8
perform_cv(nb_model, Xw_sig8, yw_sig8, 10)
#GaussianNB --> 0.593121648385

#url count and query weight
perform_cv(nb_model, X_wc, y_wc, 10)
#GaussianNB --> 0.591572491664
'''
'''
#____________________________________________
#Decision Trees
from sklearn import tree
dt_model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=6)

myList = list(range(1,20))
neighbors = []
for items in myList:
    if items%1==0:
        neighbors.append(items)
#neighbors = filter(lambda x: x % 2 != 0, myList)

cv_scores = []

for k in neighbors:
    dt_model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=k)

    scores = perform_cv(dt_model, X, y, 10)
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
#print(MSE.index(min(MSE)))

print("The optimal Depth is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Depth')
plt.ylabel('Misclassification Error')
plt.show()




#Min sig 8 
perform_cv(dt_model, X, y, 10)
#Depth 5 DecisionTreeClassifier --> 0.647765196571
#Depth 10 DecisionTreeClassifier --> 0.643892201719
#Depth 6 DecisionTreeClassifier --> 0.653686773586

#query weight 
perform_cv(dt_model, Xw, yw, 10)
#Depth 5 DecisionTreeClassifier --> 0.647627780892
#Depth 10 DecisionTreeClassifier --> 0.639469781427
#Depth 6 DecisionTreeClassifier --> 0.655410792903

#query_weight and min sig 8
perform_cv(dt_model, Xw_sig8, yw_sig8, 10)
#Depth 5 DecisionTreeClassifier --> 0.647765196571

#url count and query weight
perform_cv(dt_model, X_wc, y_wc, 10)
#Depth 5 DecisionTreeClassifier --> 0.647627780892

#____________________________________________
#K-Nearest Neighbors

#Without scaling
from sklearn.neighbors import KNeighborsClassifier

#knn_model = KNeighborsClassifier(n_neighbors=210, weights='uniform', algorithm='auto')
#Min sig 8 
#perform_cv(knn_model, X, y, 10)

#10 KNeighborsClassifier --> 0.480872257294
#20 KNeighborsClassifier --> 0.477600081703
#50 KNeighborsClassifier --> 0.473628115472
#100 KNeighborsClassifier --> 0.478250356199
#200 KNeighborsClassifier --> 0.487895402648


Without Scaling
KNeighborsClassifier --> 0.48710868341
KNeighborsClassifier --> 0.488556312019
KNeighborsClassifier --> 0.47974803023
KNeighborsClassifier --> 0.477837063505
KNeighborsClassifier --> 0.475975246089
KNeighborsClassifier --> 0.473451692128
KNeighborsClassifier --> 0.474014063975
KNeighborsClassifier --> 0.475338520423
KNeighborsClassifier --> 0.473539513614
KNeighborsClassifier --> 0.472227820925
KNeighborsClassifier --> 0.474751702647
KNeighborsClassifier --> 0.472040768916
KNeighborsClassifier --> 0.47336522536
KNeighborsClassifier --> 0.470879110519
KNeighborsClassifier --> 0.4707167167
KNeighborsClassifier --> 0.472653239071
KNeighborsClassifier --> 0.473340462612
KNeighborsClassifier --> 0.473802706505
KNeighborsClassifier --> 0.474977133347
KNeighborsClassifier --> 0.475551853797
KNeighborsClassifier --> 0.475776717939
KNeighborsClassifier --> 0.47390286255
KNeighborsClassifier --> 0.473727951566
KNeighborsClassifier --> 0.473478167026
KNeighborsClassifier --> 0.472178957166
#The optimal number of neighbors is 3

def FeatureScaler(array):
    from sklearn.preprocessing import StandardScaler
    SS = StandardScaler();
    array = SS.fit_transform(array)
    return array

X, y
#X[2:13] = FeatureScaler(X[2:13])

#query weight 
#perform_cv(knn_model, Xw, yw, 10)
#10 KNeighborsClassifier --> 0.480884749486
#20 KNeighborsClassifier --> 0.477600081703
#50 KNeighborsClassifier --> 0.473628115472
#100 KNeighborsClassifier --> 0.478250356199
#200 KNeighborsClassifier --> 0.487920387033

#query_weight and min sig 8
#perform_cv(knn_model, Xw_sig8, yw_sig8, 10)
#10 KNeighborsClassifier --> 0.480872257294

#url count and query weight
#perform_cv(knn_model, X_wc, y_wc, 10)
#10 KNeighborsClassifier --> 0.480884749486

#With Scaling






#____________________________________________
#Random Forest
from sklearn.ensemble import RandomForestClassifier
for index in range(10):
    model = RandomForestClassifier(n_estimators=50 * (index + 1), min_samples_split=2, max_features='auto',
                                       bootstrap=True)
    perform_cv(model, X, y, 10)
#RandomForestClassifier --> 0.643692114346
#RandomForestClassifier --> 0.646603146364
#RandomForestClassifier --> 0.649026822112
#RandomForestClassifier --> 0.650551010065
#RandomForestClassifier --> 0.651088088484
#RandomForestClassifier --> 0.652037662117
#RandomForestClassifier --> 0.651438021276
#RandomForestClassifier --> 0.652387491897
#RandomForestClassifier --> 0.652062772942
#RandomForestClassifier --> 0.652737199923    
    
for index in range(5):
    model = RandomForestClassifier(n_estimators=50 * (index + 1), min_samples_split=2, max_features='auto',
                                       bootstrap=True)
    perform_cv(model, Xw, yw, 10)    
        
#RandomForestClassifier --> 0.640531639622
#RandomForestClassifier --> 0.647440393291
#RandomForestClassifier --> 0.648614575079
#RandomForestClassifier --> 0.650338467968
#RandomForestClassifier --> 0.650763283687

#____________________________________________
#SVM - Linear kernel
from sklearn import svm
svm_model = svm.SVC(kernel='linear')

perform_cv(svm_model, X, y, 2)


#perform_cv(svm_model, Xw, yw, 1)
'''
#____________________________________________
#Gradient boosting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
#model = AdaBoostClassifier(n_estimators=100)

#GradientBoostingClassifier with 50 estimators
#0.656822836763


#Min sig 8
#sig 3 removed
myList = list(range(60,101))
neighbors = []
for items in myList:
    if items%1==0:
        neighbors.append(items)
#neighbors = filter(lambda x: x % 2 != 0, myList)

cv_scores = []
model = GradientBoostingClassifier(n_estimators=41, loss="exponential")
model.fit(X, y)
print("Prediction made")
y_te = model.predict(X_te)

y_pred = [int(x) for x in y_te]
print(len(y_pred))

file = open("predictions.txt", mode='w')
for item in y_pred:
    file.write(str(item))
    file.write('\n')
file.close()

file_new = open('predictions.csv', mode='w')
for new_item in y_pred:
    file_new.write(str(new_item))
    file_new.write('\n')
file_new.close()



#for k in neighbors:
 #   model = GradientBoostingClassifier(n_estimators=41, loss="exponential")
#
 #   print("depth = {0}".format(41))
  #  scores = perform_cv(model, X, y, 10)
   # cv_scores.append(scores.mean())



#MSE = [1 - x for x in cv_scores]

# determining best k
#optimal_k = neighbors[MSE.index(min(MSE))]
#print(MSE.index(min(MSE)))

#print("The optimal number of estimators is %d" % optimal_k)

# plot misclassification error vs k
#plt.plot(neighbors, MSE)
#plt.xlabel('Estimators')
#plt.ylabel('Misclassification Error')
#plt.show()



'''
#41 0.657322522893


#10 GradientBoostingClassifier --> 0.652899779513
# 50 GradientBoostingClassifier --> 0.65591090203
#100 GradientBoostingClassifier --> 0.631410039881
#200 GradientBoostingClassifier --> 0.608285345006

#url count and query weight
#perform_cv(model, X_wc, y_wc, 10)
#50 GradientBoostingClassifier --> 0.656085624144
#100 GradientBoostingClassifier --> 0.629086257955

#XGBoost

from xgboost import XGBClassifier
model = XGBClassifier()

perform_cv(model,X_wc, y_wc, 10)

#Min sig 8
#XGBClassifier --> 0.641093608728

#url count and query weight
#XGBClassifier --> 0.639457193963


#ADA boost
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100)
perform_cv(model, X, y, 10)

#Ada boost 
'''
















   
            






    
    







