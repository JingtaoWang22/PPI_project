import sys
import numpy as np
import matplotlib.pyplot as plt

from numpy import std
from numpy import mean
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.metrics import f1_score
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# get a stacking ensemble of models
def get_stacked_models():
    models = list()
    models.append(('SVM', SVC()))
    models.append(('NB', GaussianNB()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=6)))
    models.append(('DecTree', DecisionTreeClassifier(random_state=1)))
    models.append(('RF',RandomForestClassifier(n_estimators = 500)))

    #Meta classifier:
    final_model = LogisticRegression()

    model = StackingClassifier(estimators=models, final_estimator=final_model, cv=5)
    return model

#Get Sensitivity and SPecificity of the models:
def evaluate_performance(y_true,y_pred):
    tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()

    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    return sensitivity,specificity

#Plot ROC curve
def plot_curve(model,X,y,model_name,ax):
    plot_roc_curve(model,X,y,name="ROC for {}".format(model_name),alpha=0.3,lw=1,ax=ax)

#Plot comparison bar graphs
def plot_comparisons(no_aug_score,aug_score,model_names,y_label,path):
    figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

    N = 6

    ind = np.arange(N) 
    width = 0.2       
    plt.bar(ind - 0.5*width, no_aug_score, width, label='No Aug')
    plt.bar(ind + 0.5*width, aug_score, width, label='With Aug')

    plt.ylabel(y_label, fontsize = 30)
    plt.ylim((0, 0.9))
    plt_title = y_label + " of models"
    plt.title('f1 scores of models', fontsize = 30)

    plt.xticks(ind + width / 2, model_names, fontsize='xx-large')
    plt.yticks(fontsize = "xx-large")
    plt.legend(loc = "best", fontsize = 30, ncol = 2)
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    #Load_Dataset
    #Yeast Dataset:
    #Without augmentation
    print("Loading data:")
    dataset_train_loaded = np.load("yeast_train_lag30.npy", allow_pickle = True)
    dataset_test_loaded = np.load("yeast_test_lag30.npy", allow_pickle = True)
    dataset_all_loaded = np.load("yeast_all_lag30.npy", allow_pickle = True)

    #With augmentation
    dataset_train_loaded_aug = np.load("yeast_train_lag30_aug.npy", allow_pickle = True)
    dataset_test_loaded_aug = np.load("yeast_test_lag30_aug.npy", allow_pickle = True)
    dataset_all_loaded_aug = np.load("yeast_all_lag30_aug.npy", allow_pickle = True)

    print("Data loaded")
    #Define Models:
    print("Defining models:")
    models = [SVC(),GaussianNB(),KNeighborsClassifier(n_neighbors=6),DecisionTreeClassifier(random_state=1),RandomForestClassifier(n_estimators = 500)]

    stacked_models = get_stacked_models()

    models.append(stacked_models)
    print("Models defined")

    #Split dataset:
    print("Obtaining train and test data:")
    X_train = [np.hstack(e[:2]) for e in dataset_train_loaded]
    y_train = [e[2][0] for e in dataset_train_loaded]

    X_test = [np.hstack(e[:2]) for e in dataset_test_loaded]
    y_test = [e[2][0] for e in dataset_test_loaded]

    X_train_aug = [np.hstack(e[:2]) for e in dataset_train_loaded_aug]
    y_train_aug = [e[2][0] for e in dataset_train_loaded_aug]

    X_test_aug = [np.hstack(e[:2]) for e in dataset_test_loaded_aug]
    y_test_aug = [e[2][0] for e in dataset_test_loaded_aug]

    print("Data obtained. Running model:")
    #Run models and evaluate performace:
    accuracy_score,precision,f1_scores,recall_scores,sensitivity,specificity = [],[],[],[],[],[]
    accuracy_score_aug,precision_aug,f1_scores_aug,recall_scores_aug,sensitivity_aug,specificity_aug = [],[],[],[],[],[]

    model_names = ['SVM','NB','KNN','DecTree','RF','Stacked']
    i = 0
    fig,ax=plt.subplots()

    print("Without augmentation:")
    #Run models on data without augmentation:
    for model in models:
        print("{} Model training.".format(model_names[i]))
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        accuracy_score.append(model.score(X_test,y_test))
        precision.append(precision_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))

        sen,spec = evaluate_performance(y_test,y_pred)
        sensitivity.append(sen)
        specificity.append(spec)
        plot_curve(model,X_test,y_test,model_names[i],ax)
        print("Performance Evaluation complete.")
        i+=1
    
    #Redefine Models
    models = [SVC(),GaussianNB(),KNeighborsClassifier(n_neighbors=6),DecisionTreeClassifier(random_state=1),RandomForestClassifier(n_estimators = 500)]

    stacked_models = get_stacked_models()

    models.append(stacked_models)

    i = 0
    fig,ax=plt.subplots()

    print("Running models with augmentation:")
    #Run models on data with augmentation:
    for model in models:
        print("{} Model training.".format(model_names[i]))
        model.fit(X_train_aug,y_train_aug)
        y_pred = model.predict(X_test_aug)
        
        accuracy_score_aug.append(model.score(X_test_aug,y_test_aug))
        precision_aug.append(precision_score(y_test_aug, y_pred))
        f1_scores_aug.append(f1_score(y_test_aug, y_pred))
        recall_scores_aug.append(recall_score(y_test_aug, y_pred))

        sen,spec = evaluate_performance(y_test_aug,y_pred)
        sensitivity_aug.append(sen)
        specificity_aug.append(spec)
        plot_curve(model,X_test_aug,y_test_aug,model_names[i],ax)
        print("Performance Evaluation complete.")
        i+=1

    #Plot performance:
    print("Plotting comparison graphs:")
    plot_comparisons(accuracy_score,accuracy_score_aug,model_names,"Accuracy","accuracy.svg")
    plot_comparisons(precision,precision_aug,model_names,"Precision","precision.svg")
    plot_comparisons(f1_scores,f1_scores_aug,model_names,"f1 scores","f1_scores.svg")
    plot_comparisons(recall_scores,recall_scores_aug,model_names,"Recall","recall.svg")
    plot_comparisons(sensitivity,sensitivity_aug,model_names,"Sensitivity","sensitivity.svg")
    plot_comparisons(specificity,specificity_aug,model_names,"Specificity","specificity.svg")

    print("Run complete")