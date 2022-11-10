
from decimal import Rounded
from math import nan
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

data = pd.read_csv('house-votes-84.csv')
data.columns = ['Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
                'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']


# replace y with 1 and n with 0

datanew = data.replace(['y', 'n'], [1, 0])
dataz = datanew.replace(['?'], [nan])

# replace all the missing values with majority value
# print(dataz.mode().iloc[0])


result = dataz.fillna(dataz.mode().iloc[0])


# split the data into training and testing data
X = result.drop(['Name'], axis=1)  # features
y = result['Name']  # target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2)  # split the data into training and testing data

# model = DecisionTreeClassifier()  # create a model
# model.fit(X_train, y_train)  # train the model

# tree.export_graphviz(model, out_file='tree7.dot', feature_names=['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
#                                                                  'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'],
#                      class_names=result['Name'],
#                      label='all', rounded=True, filled=True)  # export the model
# predictions = model.predict(X_test)  # predict the test data
# # print the accuracy by comparing the test data and predictions
# accuracy = accuracy_score(y_test, predictions)
# # print size of tree
# print("Tree size = ", model.tree_.node_count)
# print("accuracy of the model is = ", accuracy)



#for loop for models with different values of k
values = [10,20,30,40,50,60,70,80,90,100]
for i in values:
    print("test data = ",i)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=i)  # split the data into training and testing data

    model = DecisionTreeClassifier()  # create a model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)  # predict the test data
    # print the accuracy by comparing the test data and predictions
    accuracy = accuracy_score(y_test, predictions)
    # print size of tree
    print("Tree size = ", model.tree_.node_count)
    print("accuracy of the model is = ", accuracy)
    print("\n")
    #graph for each k
    tree.export_graphviz(model, out_file='tree'+str(i)+'.dot', feature_names=['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
                                                                 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'],
                        label='all', rounded=True, filled=True)
    # print("\n")   
