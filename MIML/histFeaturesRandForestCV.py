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

def zeroFun(num, vect):
    if vect == [0]*len(vect):
        return 0.0
    else:
        return num
   
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
        ypred.append(zeroFun(clf.predict_proba(XCV)[0][1], XCV))
    sys.stdout.write("\r")
    fpr, tpr, thresholds = metrics.roc_curve(y, ypred, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return (auc, ypred)

def trainOneClass(X, y):
    ns = [100]
    ds = [ 20]
    
    scores = []
    scoremax = -1
    
    for n in ns:
        for d in ds:
            (score, pred) = calcAUCScoreCV(X, y, n, d)
            
            scores.append(scores)
            
            if score > scoremax:
                predmax  = pred
                nmax     = n
                dmax     = d
                scoremax = score
            
            print "  Params: n = ", n, " d = ", d, "--> Score: ", score
            
    return (nmax, dmax, predmax)

def printY(y, ids, name):
    file = open(name + ".csv", "w")
    file.write("Id,Probability\n")
    
    for i in range(len(ids)):
        for k in range(len(y)):
            file.write(str(ids[i]*100+k)+ "," + str(y[k][i]) + "\n")
            
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
    (Xtrain, ytrain, Xtest, idtrain, idtest) = processAndGetData()
    
    Ztrain = load2DZ("Z6train")
    Ztest  = load2DZ("Z6test")
    
    ypred = []
    yreal = []
    
    ytest = []
    for k in range(19):
        print "Class: ", k
        (n, d, yret) = trainOneClass(Ztrain, ytrain[k])
        # n = 100
        # d = 20
        
        ypred = ypred + yret
        yreal = yreal + ytrain[k]
                
        clf = RandomForestClassifier(n_estimators = n, max_depth = d)
        clf.fit(Ztrain, ytrain[k])
        y = clf.predict_proba(Ztest)
        
        y1 = [zeroFun(y[i][1], Ztest[i]) for i in range(len(y))]
        ytest.append(y1)
        
    fpr, tpr, thresholds = metrics.roc_curve(yreal, ypred, pos_label=1)
    print metrics.auc(fpr,tpr)
    printY(ytest, idtest, "youtRFMIML-knn4")
    
if __name__ == "__main__":
    main()