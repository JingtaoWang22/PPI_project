import sys
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# Encoding used by Guo et al. 2008 https://academic.oup.com/nar/article/36/9/3025/1104168?login=true

# Physiochemical properties
# Numpy array; rows ordered by AA orders (index 0 for A, 1 for C, etc.);
# Columns: Hydrophobicity; hydrophilicity; volumn of side chians; polarity; polarizability; 
#           Solvent accessible surface area; Net charge index of side chains
properties = np.array([[0.62, -0.5, 27.5, 8.1, 0.046, 1.181, 0.007187] , 
                       [0.29, -1.0, 44.6, 5.5, 0.128, 1.461, -0.03661] , 
                       [-0.9, 3.0, 40.0, 13.0, 0.105, 1.587, -0.02382] , 
                       [-0.74, 3.0, 62.0, 12.3, 0.151, 1.862, 0.006802] , 
                       [1.19, -2.5, 115.5, 5.2, 0.29, 2.228, 0.037552] , 
                       [0.48, 0.0, 0.0, 9.0, 0.0, 0.881, 0.179052] , 
                       [-0.4, -0.5, 79.0, 10.4, 0.23, 2.025, -0.01069] , 
                       [1.38, -1.8, 93.5, 5.2, 0.186, 1.81, 0.021631] , 
                       [-1.5, 3.0, 100.0, 11.3, 0.219, 2.258, 0.017708] , 
                       [1.06, -1.8, 93.5, 4.9, 0.186, 1.931, 0.051672] , 
                       [0.64, -1.3, 94.1, 5.7, 0.221, 2.034, 0.002683] , 
                       [-0.78, 2.0, 58.7, 11.6, 0.134, 1.655, 0.005392] , 
                       [0.12, 0.0, 41.9, 8.0, 0.131, 1.468, 0.239531] , 
                       [-0.85, 0.2, 80.7, 10.5, 0.18, 1.932, 0.049211] , 
                       [-2.53, 3.0, 105.0, 10.5, 0.291, 2.56, 0.043587] , 
                       [-0.18, 0.3, 29.3, 9.2, 0.062, 1.298, 0.004627] , 
                       [-0.05, -0.4, 51.3, 8.6, 0.108, 1.525, 0.003352] , 
                       [1.08, -1.5, 71.5, 5.9, 0.14, 1.645, 0.057004] , 
                       [0.81, -3.4, 145.5, 5.4, 0.409, 2.663, 0.037977] , 
                       [0.26, -2.3, 117.3, 6.2, 0.298, 2.368, 0.023599]])

# Normalize the property matrix with mean=0 and corresponding stds
def normalize_property_matrix(m):
  pjs = np.mean(m, axis = 0)
  sjs = np.std(m, axis = 0)
  normalized = np.zeros((len(m), len(m[0])))
  for i in range(len(m)):
    for j in range(len(m[0])):
      normalized[i, j] = (m[i, j] - pjs[j]) / sjs[j]
  
  return normalized


# Each AC[lag, j] value, according to eqn 2 in paper
def calc_ac_entry2(lag, x):
  n = len(x)
  meanj = np.mean(x, axis = 0)
  #print(meanj)
  ac = np.mean([(x[i] - meanj) * (x[i + lag] - meanj) for i in range(n - lag)], axis = 0)
  #print(ac, x[0] - meanj)

  return ac



# Wrapper function for the whole AC matrix for each seq's properties

def calc_ac2(x, max_lag, num_p = 7):
  D = np.zeros((max_lag, num_p))
  for i in range(max_lag):
    D[i] = calc_ac_entry2(i + 1, x)
  
  return D


# Get the physiochemical properties of a seq
def seq_properties(s, p):
  x = np.zeros((len(s), 7))
  for i in range(len(s)):
    #for j in range(7):
      #x[i, j] = p[s[i], j]
    x[i] = p[s[i]]
  
  
  return x


# Encode each X
# seq s1 (list of aa indexes) -> properties matrix -> matrix D1
# seq s2 (list of aa indexes) -> properties matrix -> matrix D2
# Returns falttened + concatenated D1 and D2
def encoding(s1, s2, p, max_lag = 20, num_p = 7):
  x1 = seq_properties(s1, p)
  x2 = seq_properties(s2, p)
  D1 = calc_ac2(x1, max_lag, num_p)
  D2 = calc_ac2(x2, max_lag, num_p)
  D1 = D1.reshape(-1)
  D2 = D2.reshape(-1)
  #x_conc = np.hstack((D1, D2))

  return D1, D2



# Preprocessing
def shuffle_dataset(dataset, seed):
  np.random.seed(seed)
  np.random.shuffle(dataset)
  return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

norm_properties = normalize_property_matrix(properties)

aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
           'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}


##################################################################################
##############################PREPROCESSING YEAST DATASET#########################
##################################################################################
## reading dataset, modified from Jingtao's code
# protein dictionary
dic_file=open('yeast_dic.tsv','r')
dic_lines=dic_file.readlines()
dic={}
for i in dic_lines:
  item=i.strip().split()
  dic[item[0]]=item[1]
dic_file.close()

#PPIs
ppi_file=open('yeast_ppi.tsv','r')
ppi_lines=ppi_file.readlines()

## constructing dataset
#3-mers not used because not necessarily good for ML models
aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
           'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

dataset = []
for i in ppi_lines:
  ppi = i.strip().split()
  p1, p2, interaction = dic[ppi[0]], dic[ppi[1]], [int(ppi[2])]

  p1_list = [aa_dict[elem] for elem in p1]

  p2_list = [aa_dict[elem] for elem in p2]

  #Properties representation encoding
  D1, D2 = encoding(p1_list, p2_list, norm_properties, max_lag = 30, num_p = 7)

  dataset.append([D1, D2, interaction])


dataset = shuffle_dataset(dataset, 1234)
dataset_train, dataset_test = split_dataset(dataset, 0.8)
print("Preprocessing completed. number of training samples = {}, number testing samples = {}".format(len(dataset_train), len(dataset_test)))
print("Print out train sample #0: ", dataset_train[0])

np.save("yeast_train.npy", dataset_train)
np.save("yeast_test.npy", dataset_test)
np.save("yeast_all.npy", dataset)



##################################################################################
#########################TUNE YEAST DATASET PARAMETERS############################
##################################################################################
################### Tune for "lag" values ############################
# The difference between this block and the earlier block on preprocessing is
#  that this block runs lag=40.
#  the first l * 7 entree for each encoding is encoding for lag = l
# Preprocessing
def shuffle_dataset(dataset, seed):
  np.random.seed(seed)
  np.random.shuffle(dataset)
  return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

## reading dataset, modified from Jingtao's code
# protein dictionary
dic_file=open('yeast_dic.tsv','r')
dic_lines=dic_file.readlines()
dic={}
for i in dic_lines:
  item=i.strip().split()
  dic[item[0]]=item[1]
dic_file.close()

#PPIs
ppi_file=open('yeast_ppi.tsv','r')
ppi_lines=ppi_file.readlines()

## constructing dataset
#3-mers not used because not necessarily good for ML models
aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
           'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

dataset_large = []
for i in ppi_lines:
  ppi = i.strip().split()
  p1, p2, interaction = dic[ppi[0]], dic[ppi[1]], [int(ppi[2])]

  p1_list = [aa_dict[elem] for elem in p1]

  p2_list = [aa_dict[elem] for elem in p2]

  #Properties representation encoding
  D1, D2 = encoding(p1_list, p2_list, norm_properties, max_lag = 40, num_p = 7)

  dataset_large.append([D1, D2, interaction])


dataset_large = shuffle_dataset(dataset_large, 1234)
dataset_train_large, dataset_test_large = split_dataset(dataset_large, 0.8)
print("Preprocessing completed. number of training samples = {}, number testing samples = {}".format(len(dataset_train_large), len(dataset_test_large)))
print("Print out train sample #0: ", dataset_train_large[0])


np.save("yeast_train_large.npy", dataset_train_large)
np.save("yeast_test_large.npy", dataset_test_large)
np.save("yeast_all_large.npy", dataset_large)


dataset_train_large_loaded = np.load("yeast_train_large.npy", allow_pickle = True)
dataset_test_large_loaded = np.load("yeast_test_large.npy", allow_pickle = True)
dataset_all_large_loaded = np.load("yeast_all_large.npy", allow_pickle = True)

# See results of different lag values using classifiers same as above
def svm_score(X, y, testX, testy):
  clf = SVC()
  clf.fit(X, y)
  score = clf.score(testX, testy)

  return score


def knn_score(X, y, testX, testy):
  knn_clf = KNeighborsClassifier(n_neighbors=5)
  knn_clf.fit(X, y)
  score = knn_clf.score(testX, testy)

  return score


def rf_score(X, y, testX, testy, n_est = 500):
  rf_clf = RandomForestClassifier(n_estimators = n_est)
  rf_clf.fit(X, y)
  score = rf_clf.score(testX, testy)

  return score


def score_lags(dataset_train_loaded, dataset_test_loaded, dataset_all_loaded, lags, save_dataset = False):
  y = [e[2][0] for e in dataset_train_loaded]
  testy = [e[2][0] for e in dataset_test_loaded]
  svm_scores, knn_scores, rf_scores = [], [], []
  for i in range(len(lags)):
    l = lags[i]
    X = [np.hstack((e[0][:(l*7)], e[1][:(l*7)])) for e in dataset_train_loaded]
    testX = [np.hstack((e[0][:(l*7)], e[1][:(l*7)])) for e in dataset_test_loaded]
    svm_scores.append(svm_score(X, y, testX, testy))
    knn_scores.append(knn_score(X, y, testX, testy))
    rf_scores.append(rf_score(X, y, testX, testy))

    if save_dataset:
      savetrain = [[e[0][:(l*7)], e[1][:(l*7)], e[2]] for e in dataset_train_loaded]
      savetest = [[e[0][:(l*7)], e[1][:(l*7)], e[2]] for e in dataset_test_loaded]
      saveall = [[e[0][:(l*7)], e[1][:(l*7)], e[2]] for e in dataset_all_loaded]
      np.save("yeast_train_lag={}.npy".format(l), savetrain)
      np.save("yeast_test_lag={}.npy".format(l), savetest)
      np.save("yeast_all_lag={}.npy".format(l), saveall)

  
  return svm_scores, knn_scores, rf_scores


lags = [10, 15, 20, 25, 30, 35, 40]
svm_scores, knn_scores, rf_scores = score_lags(dataset_train_large_loaded, dataset_test_large_loaded, dataset_all_large_loaded, lags, save_dataset = True)
print(svm_scores, knn_scores, rf_scores)

plt.plot(lags, svm_scores, marker = "o", label = "SVM")
plt.plot(lags, knn_scores, marker = "o", label = "KNN")
plt.plot(lags, rf_scores, marker = "o", label = "RF")
plt.xlabel("#Lags")
plt.ylabel("Accuracies")
plt.legend()
plt.show()



##################################################################################
###################################DATA AUGMENTATION##############################
##################################################################################
load30train = np.load("yeast_train_lag30.npy", allow_pickle = True)
load30test = np.load("yeast_test_lag30.npy", allow_pickle = True)
load30all = np.load("yeast_all_lag30.npy", allow_pickle = True)

def data_augmentaton(dataset, savedir):
  new_dataset = []
  for i in range(len(dataset)):
    new_dataset.append(dataset[i])
    new_dataset.append([dataset[i][1], dataset[i][0], dataset[i][2]])
  
  np.random.shuffle(new_dataset)
  np.save(savedir, new_dataset)

  return

data_augmentaton(load30train, "yeast_train_lag30_aug.npy")
print("Completed")


##################################################################################
###################################HUMAN DATASET##################################
##################################################################################
## reading dataset, modified from Jingtao's code
# protein dictionary
dic_file = open("human_dic.txt", "r")
dic_lines=dic_file.readlines()
dic={}
for i in dic_lines:
  item=i.strip().split()
  dic[item[0]]=item[1]
dic_file.close()

#PPIs
ppi_file = open("human_ppi.txt", "r")
ppi_lines=ppi_file.readlines()

dataset = []

not_aa = ["B", "J", "O", "U", "X", "Z"]

for i in range(len(ppi_lines)):
  ppi = ppi_lines[i].strip().split()
  p1, p2, interaction = dic[ppi[0]], dic[ppi[1]], [int(ppi[2])]

  valid = not(any(elem in not_aa  for elem in p1) or any(elem in not_aa  for elem in p2))

  if len(p1) > 31 and len(p2) > 31 and valid:

    p1_list = [aa_dict[elem] for elem in p1]

    p2_list = [aa_dict[elem] for elem in p2]

    #Properties representation encoding
    D1, D2 = encoding(p1_list, p2_list, norm_properties, max_lag = 30, num_p = 7)
  
  if i % 5000 == 0:
    print("finished line {}".format(i))

  dataset.append([D1, D2, interaction])


dataset = shuffle_dataset(dataset, 1234)
dataset_train, dataset_test = split_dataset(dataset, 0.8)
print("Preprocessing completed. number of training samples = {}, number testing samples = {}".format(len(dataset_train), len(dataset_test)))
print("Print out train sample #0: ", dataset_train[0])
np.save("human_train_lag30.npy", dataset_train)
np.save("human_test_lag30.npy", dataset_test)
np.save("human_all_lag30.npy", dataset)
