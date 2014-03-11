from string               import split
from sklearn.metrics      import auc_score
from sklearn              import svm
from numpy                import array
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search  import GridSearchCV
from numpy                import array
from sklearn              import cross_validation, metrics
from sklearn.ensemble     import RandomForestClassifier 

import sys



def processAndGetData():
    hist_file   = open('C:\Users\Bojan\Desktop\ML\Kaggle\Birds\mlsp_contest_dataseta\supplemental_data\\histogram_of_segments.txt', 'r')
    labels_file = open('C:\Users\Bojan\Desktop\ML\Kaggle\Birds\mlsp_contest_dataseta\essential_data\\rec_labels_test_hidden.txt', 'r')
    
    Xtrain   = []
    idtrain  = []
    Xtest  = []
    idtest = []
    ytrain = [[] for i in range(19)]
    
    hist_line  = hist_file.readline().split(',')
    label_line = labels_file.readline().split(',')
    
    label_line = labels_file.readline().split(',')
    hist_line  = hist_file.readline().split(',')[1:]
    
    while len(label_line[0]) > 0:
        
        features   = [float(hist_line[i]) for i in range(len(hist_line))]
        id         = int(label_line[0])
        
        if len(label_line) > 1 and '?' in label_line[1]:
            idtest.append(id)
            Xtest.append(features)
        else:
            labels = [int(label_line[i]) for i in range(1, len(label_line))]
            
            idtrain.append(id)
            Xtrain.append(features)
            
            for i in range(19):
                ytrain[i].append(int(i in labels))
    
        label_line = labels_file.readline().split(',')
        hist_line  = hist_file.readline().split(',')[1:]
        
    hist_file.close()
    labels_file.close()
    
    return (Xtrain, ytrain, Xtest, idtrain, idtest)
    
def calcAUCScoreCV(X, y, n, d):
    ypred = []
    
    for i in range(len(X)):
        
        sys.stdout.write("\r")
        sys.stdout.write("   Run: %3d/%d" % (i, len(X)))
        sys.stdout.flush()
        
        Xtrain = X[0:i] + X[i+1:]
        ytrain = y[0:i] + y[i+1:]
        
        XCV = X[i]
        yCV = y[i]
        
        clf = RandomForestClassifier(n_estimators = n, max_depth = d)
        clf.fit(Xtrain, ytrain)
        ypred.append(clf.predict_proba(XCV)[0][1])
    
    sys.stdout.write("\r")
    fpr, tpr, thresholds = metrics.roc_curve(y, ypred, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    
    return auc

def trainOneClass(X, y):
    ns = [50, 100, 200]
    ds = [7, 15, 30]
    
    scores = []
    scoremax = -1
    
    for n in ns:
        for d in ds:
            score = calcAUCScoreCV(X, y, n, d)
            
            scores.append(scores)
            
            if score > scoremax:
                nmax     = n
                dmax     = d
                scoremax = score
            
            print "  Params: n = ", n, " d = ", d, "--> Score: ", score
            
    return (nmax, dmax)

def printY(y, ids, name):
    file = open(name + ".csv", "w")
    file.write("Id,Probability\n")
    
    for i in range(len(ids)):
        for k in range(len(y)):
            file.write(str(ids[i]*100+k)+ "," + str(y[k][i]) + "\n")
            
    file.close()
    
def main():
    (Xtrain, ytrain, Xtest, idtrain, idtest) = processAndGetData()
    
    ytest = []
    for k in range(19):
        print "Class: ", k
        (n, d) = trainOneClass(Xtrain, ytrain[k])
        print "\n  Params chosen: n = ", n, " d = ", d
        
        clf = RandomForestClassifier(n_estimators = n, max_depth = d)
        clf.fit(Xtrain, ytrain[k])
        y = clf.predict_proba(Xtest)
        
        y1 = [y[i][1] for i in range(len(y))]
        ytest.append(y1)
        
    printY(ytest, idtest, "youtRF")
    
if __name__ == "__main__":
    main()