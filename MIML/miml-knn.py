from string                 import split
from sklearn.metrics        import auc_score
from sklearn                import svm
from numpy                  import array
from sklearn.linear_model   import LogisticRegression
from sklearn.grid_search    import GridSearchCV
from numpy                  import array
from sklearn                import cross_validation, metrics
from sklearn.ensemble       import RandomForestClassifier 
from scipy.spatial.distance import euclidean
from random                 import random
from math                   import sqrt

import sys



def processAndGetData():
    seg_file    = open('C:\Users\Bojan\Desktop\ML\Kaggle\Birds\mlsp_contest_dataseta\supplemental_data\\segment_features.txt', 'r')
    labels_file = open('C:\Users\Bojan\Desktop\ML\Kaggle\Birds\mlsp_contest_dataseta\essential_data\\rec_labels_test_hidden.txt', 'r')
    
    
    
    Xtrain      = []
    idtrain     = []
    Xtest       = []
    idtest      = []
    ytrain      = [[] for i in range(19)]
    labelstrain = []
    
    seg_line   = seg_file.readline().split(',')
    label_line = labels_file.readline().split(',')
    
    label_line = labels_file.readline().split(',')
    seg_line   = seg_file.readline().split(',')
    id_seg     = int(seg_line[0])
    
    n_sum = len(seg_line) - 2
    n_feat = 0
    sum_feat   = [0] * n_sum
    sumsq_feat = [0] * n_sum
    
    
    while len(label_line[0]) > 0:
        id_label   = int(label_line[0])
        
        if len(label_line) > 1 and '?' in label_line[1]:
            idtest.append(id_label)
            Xtest.append([])
            
            while id_seg == id_label:
                features = [float(seg_line[i]) for i in range(2, len(seg_line))]
                
                sum_feat   = [sum_feat[i]  +features[i]    for i in range(len(sum_feat))]
                sumsq_feat = [sumsq_feat[i]+features[i]**2 for i in range(len(sum_feat))]
                n_feat     += 1
                
                Xtest[-1].append(features)
                
                seg_line  = seg_file.readline().split(',')
                if len(seg_line) > 1:
                    id_seg   = int(seg_line[0])
                else:
                    id_seg = -1
                    
        else:
            labels = [int(label_line[i]) for i in range(1, len(label_line))]
            
            idtrain.append(id_label)
            labelstrain.append(labels)
            Xtrain.append([])
            
            while id_seg == id_label:
                features = [float(seg_line[i]) for i in range(2, len(seg_line))]
                
                sum_feat   = [sum_feat[i]  +features[i]    for i in range(len(sum_feat))]
                sumsq_feat = [sumsq_feat[i]+features[i]**2 for i in range(len(sum_feat))]
                n_feat     += 1
                
                Xtrain[-1].append(features)
                
                seg_line  = seg_file.readline().split(',')
                if len(seg_line) > 1:
                    id_seg   = int(seg_line[0])
                else:
                    id_seg = -1
            
            for i in range(19):
                ytrain[i].append(int(i in labels))

        label_line = labels_file.readline().split(',')
        
    seg_file.close()
    labels_file.close()
    
    sum_feat   = [sum_feat[i]/n_feat                          for i in range(len(sum_feat))]
    sumsq_feat = [sqrt(sumsq_feat[i]/n_feat - sum_feat[i]**2) for i in range(len(sum_feat))]
    
    print sum_feat
    print sumsq_feat
    
    # print idtrain
    return (Xtrain, ytrain, Xtest, idtrain, idtest, labelstrain, sum_feat, sumsq_feat)
  
def normalizeList(l, mean, std):
    return [(l[i]-mean[i])/std[i] for i in range(len(l))]
  
  
def calcHausDist(bag1, bag2, mean, std):
    n1 = len(bag1)
    n2 = len(bag2)
    # if n1 == 0 and n2 == 0:
    #     return 0.0
    if n1 == 0 or  n2 == 0:
        return float("+inf")
        
    total_dist = 0.0
    
    for a in bag1:
        minDist = float("+inf")
        for b in bag2:
            dist = euclidean(normalizeList(a, mean, std), normalizeList(b, mean, std))
            if dist < minDist:
                minDist = dist
        total_dist += minDist
        
    for b in bag2:
        minDist = float("+inf")
        for a in bag1:
            dist = euclidean(normalizeList(b, mean, std), normalizeList(a, mean, std))
            if dist < minDist:
                minDist = dist
        total_dist += minDist
    
    return total_dist / (n1+n2)

def sortBags(dist, idx):
    n = len(dist)
    
    for i in range(n-1):
        for j in range(i+1, n):
            if dist[i] > dist[j] or (dist[i] == dist[j] and random() < 0.5):
                t = dist[i]
                dist[i] = dist[j]
                dist[j] = t
                t = idx[i]
                idx[i] = idx[j]
                idx[j] = t
    return idx 

def Ztransform(X, labels, ntrain, k, c, mean, std):
    n = len(X)
    Z = [[0]*19 for i in range(n)]
    
    for i in range(n):
        sys.stdout.write("\r")
        sys.stdout.write("   Run: %3d/%d" % (i, n))
        sys.stdout.flush()
        Haus_dist = []
        bag_num   = []
        for j in range(n):
            if i != j:
                dist = calcHausDist(X[i], X[j], mean, std)
                if dist < float("+inf"):
                    Haus_dist.append(dist)
                    bag_num.append(j)
                    
        bag_num = sortBags(Haus_dist, bag_num)
        
        count = 0
        idx = 0
        while count < k:
            if len(bag_num) <= idx:
                break
            bag = bag_num[idx]
            if bag < ntrain:
                for l in labels[bag]:
                    Z[i][l] += 1
                count += 1
            idx += 1
            
            
        count = 0
        idx = 0
        if i < ntrain:
            while count < c:
                if len(bag_num) <= idx:
                    break
                bag = bag_num[idx]
                if bag < ntrain:
                    for l in labels[i]:
                        Z[bag][l] += 1
                    count += 1
                idx += 1
    
    # print2DZ(Ztrain, "Z")
    return (Z[0:ntrain], Z[ntrain:])

def calcAUCScoreCV(X, y, n, d):
    ypred = []
    
    for i in range(len(X)):
        Xtrain = X[0:i] + X[i+1:]
        ytrain = y[0:i] + y[i+1:]
        
        XCV = X[i]
        yCV = y[i]
        
        clf = RandomForestClassifier(n_estimators = n, max_depth = d)
        clf.fit(Xtrain, ytrain)
        # print clf.predict(XCV)
        ypred.append(int(clf.predict(XCV)[0]))
    
    # print "    ", y
    # print "    ", ypred
    fpr, tpr, thresholds = metrics.roc_curve(y, ypred, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc

def trainOneClass(X, y):
    # return (100, 5)
    
    # ns = [40, 100, 200, 300]
    # ds = [10, 20, 40, 60]
    
    ns = [100]
    ds = [ 20]
    
    scores = []
    scoremax = -1
    
    for n in ns:
        for d in ds:
        
            score = calcAUCScoreCV(X, y, n, d)
            print "    Now testing: ", n, d, score
            scores.append(scores)
            
            if score > scoremax:
                nmax = n
                dmax = d
                scoremax = score
                
    return (nmax, dmax)

def printY(y, ids, name):
    file = open(name + ".csv", "w")
    file.write("Id,Probability\n")
    
    for i in range(len(ids)):
        for k in range(len(y)):
            file.write(str(ids[i]*100+k)+ "," + str(y[k][i]) + "\n")
            
    file.close()
    
def print2DZ(array, name):
    file = open(name + ".txt", "w")
    
    for i in range(len(array)):
        for k in range(len(array[i])):
            file.write(str(array[i][k])+ " ")
        file.write("\n")
            
    file.close()
    
def load2DZ(name):
    ret = []
    file = open(name + ".txt")
    
    for line in file:
        if len(line) > 1:
            s = line.strip('\n').strip().split(" ")
            ret.append([int(ss) for ss in s])
            
    file.close()
    return ret
    
def main():
    (Xtrain, ytrain, Xtest, idtrain, idtest, labels, mean, std) = processAndGetData()
    
    (Ztrain, Ztest) = Ztransform(Xtrain+Xtest, labels, len(Xtrain), 8, 7, mean, std)
    print2DZ(Ztrain, "Z6train")
    print2DZ(Ztest , "Z6test")
    
    
    
    # Ztest  = ZtransformTest(Xtest, Xtrain)
    
if __name__ == "__main__":
    main()