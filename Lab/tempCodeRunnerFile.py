modelKNN = KNeighborsClassifier(n_neighbors=k[5], p=2)
# modelKNN.fit(X_train,Y_train)
# # Model Testing
# Y_pred= modelKNN.predict(X_test)
# KNNScore = accuracy_score(Y_test, Y_pred)

# print(KNNScore)
# print('Confusion Matrix: ')
# print(confusion_matrix(Y_test, Y_pred))
# print('Classification Report: ')
# print(classification_report(Y_test, Y_pred))
# # 7.2(B)
# # Decision Tree parameter
# ASM_function = ["entropy", "gini"]
# maxD = [4, 5, 6, None] # try at least 2 values
# # Model Training

# for i in ASM_function:
#     for j in maxD:
#         ModelDT = DecisionTreeClassifier(criterion=i, splitter='best',max_depth = j )
#         ModelDT.fit(X_train,Y_train)
#         # Model Testing
#         Y_pred= ModelDT.predict(X_test)
#         DTScore = accuracy_score(Y_test, Y_pred)
#         print("criterion = ",i,"max_depth =",j)
#         print(DTScore)
# # Print Confusion Matrix and Classification Report for best k
# # option 1 criterion =  entropy max_depth = 4
# # option 2 criterion =  entropy max_depth = 6 , criterion =  gini max_depth = 5
# ModelDT = DecisionTreeClassifier(criterion="entropy", splitter='best',max_depth = 6 )
# ModelDT.fit(X_train,Y_train)
# # Model Testing
# Y_pred= ModelDT.predict(X_test)
# DTScore = accuracy_score(Y_test, Y_pred)
# print("criterion = ",i,"max_depth =",j)
# print(DTScore)
# print('Confusion Matrix: ')
# print(confusion_matrix(Y_test, Y_pred))
# print('Classification Report: ')
# print(classification_report(Y_test, Y_pred))
# # Visualize Decision Tree

# feature_names = X_train.columns
# Labels = str(np.unique(Y_train))
# # print(Labels)
# # tree.plot_tree( ModelDT,feature_names = feature_names,class_names = Labels,rounded = True,filled = True, fontsize=9)
# # plt.show()

# # 7.2(C)
# # Random Forest parameter
# ASM_function = ['entropy', 'gini']
# nEstimator = 100
# nJob = 2
# rState = 10
# # for i in ASM_function :
# #     # Model Training
# #     RandomF = RandomForestClassifier(criterion=i,n_estimators=nEstimator, n_jobs=nJob, random_state=rState)
# #     RandomF.fit(X_train,Y_train)
# #     # Model Testing
# #     Y_pred= RandomF.predict(X_test)
# #     RFScore = accuracy_score(Y_test, Y_pred)
# #     print("criterion = ",i)
# #     print(RFScore)
# # option 1 criterion = เท่ากัน
# # option 2 criterion = entropy
# # Print Confusion Matrix and Classification Report for best k
# # Model Training
# RandomF = RandomForestClassifier(criterion="entropy",n_estimators=nEstimator, n_jobs=nJob, random_state=rState)
# RandomF.fit(X_train,Y_train)
# # Model Testing
# Y_pred= RandomF.predict(X_test)
# RFScore = accuracy_score(Y_test, Y_pred)
# print(RFScore)
# print('Confusion Matrix: ')
# print(confusion_matrix(Y_test, Y_pred))
# print('Classification Report: ')
# print(classification_report(Y_test, Y_pred))
# # Visualize Feature Important Score
# feature_imp = pd.Series(RandomF.feature_importances_, index = feature_names).sort_values(ascending=False)

# # Creating a bar plot
# # plt.figure(figsize=(15,15))
# # sns.barplot(x=feature_imp, y=feature_imp.index)
# # plt.show()

# # Visualize selected estimator [0-5] tree structure of Random forest
# # fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=150)
# # for index in range(0, 5):
# #     tree.plot_tree( RandomF.estimators_[index],feature_names = feature_names,class_names= Labels,filled = True,ax = axes[index])
# #     axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
# # plt.show()

# # Create Model List
# classification = { "KNN": KNeighborsClassifier(), "DT": DecisionTreeClassifier(), "RF": RandomForestClassifier() }
# # Create Parameter Dictionary for KNN
# K_list = [1, 3, 5, 7, 9 , 11, 13, 15, 17, 19, 21, 23, 25, 35, 45]
# KNN_param = dict(n_neighbors=K_list)
# # Create Parameter Dictionary for Decision Tree
# ASM_function = ["entropy", "gini"]
# maxD = [ 4, 5, 6, None]
# maxF = ["auto", "log2", None]
# minSample = [1,2, 4]
# DT_param= dict(criterion=ASM_function, max_depth = maxD, min_samples_leaf = minSample, max_features = maxF)
# # Create Parameter Dictionary for Random Forest (including same parameters as Decision Tree)
# nEst = [10, 30, 50, 100]
# RF_param = dict(n_estimators = nEst, criterion=ASM_function, max_depth = maxD, min_samples_leaf = minSample,max_features = maxF)
# # Perform GridsearchCV() for each classification model
# #grid = GridSearchCV(estimator = model,n_jobs = 1,verbose = 10,scoring = "accuracy",cv = 2,param_grid)
# #grid_result = grid.fit(X_train, Y_train)