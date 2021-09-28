# At this point I run the XGBoost models.
# The first is a simple iteration for the target variable Z (which only has two classes). 
# The second is the multiclass model that predicts on all the different labels of prognosis

# Looking at the performance metrics, both models do about as perfect as possible. 
# That is not that surprising given the order and uniform distribution of the data. 

# More interestingly, this was more for the sake of my own learning and figuring out how to use multiclass classificaiton.
# The differences between multiclass and binary became more evident to me when writing the code for ascertaining performance metrics.
# Especially as you have to specify the parameter "average=" to get appropriate precision, recall, and F1-scores



xgboost = XGBClassifier(random_state=0)
xgboost.fit(X_train, z_train)
predictions = xgboost.predict(X_test)


accuracy = accuracy_score(z_test, predictions) 
auc = roc_auc_score(z_test, predictions)

print(confusion_matrix(z_test, predictions)) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("AUC: %.2f%%" % (auc * 100.0))
print("Precision:", precision_score(z_test, predictions))
print("Recall:", recall_score(z_test, predictions))
print("F1 score:",f1_score(z_test, predictions))



xgboost = XGBClassifier(random_state=0, objective='multi:softmax') # Here I include the line "objective='multi:softmax'" to specify a multiclass iteration of the model
xgboost.fit(X_train, y_train)

predictions = xgboost.predict(X_test)

accuracy = accuracy_score(y_test, predictions) 
auc = roc_auc_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision:", precision_score(y_test, predictions, average='micro'))
print("Recall:",recall_score(y_test, predictions, average='macro'))
print("F1 score:",f1_score(y_test, predictions, average='weighted'))
