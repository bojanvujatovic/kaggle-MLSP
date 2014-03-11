from string               import split
from sklearn.metrics      import auc_score
from sklearn              import svm
from numpy                import array
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search  import GridSearchCV
from numpy                import array
from sklearn              import cross_validation, metrics



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
    
def calcAUCScoreCV(X, y, c):
    ypred = []
    
    for i in range(len(X)):
        Xtrain = X[0:i] + X[i+1:]
        ytrain = y[0:i] + y[i+1:]
        
        XCV = X[i]
        yCV = y[i]
        
        clf = svm.SVC(C = c, kernel = 'rbf', class_weight='auto')
        clf.fit(Xtrain, ytrain)
        # print clf.predict(XCV)
        ypred.append(int(clf.predict(XCV)[0]))
    
    # print "    ", y
    # print "    ", ypred
    fpr, tpr, thresholds = metrics.roc_curve(y, ypred, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc

def trainOneClass(X, y):
    Cs = [0.00005*(10**i) for i in range(4,7)] + [0.00001*(10**i) for i in range(4,7)]
    # return 0.001
    scores = []
    scoremax = -1
    
    for C in Cs:
        
        score = calcAUCScoreCV(X, y, C)
        print "    Now testing: ", C, score
        scores.append(scores)
        
        if score > scoremax:
            Cmax     = C
            scoremax = score
        
        # print C, score
        
    # print "  Max: ", Cmax, scoremax
    return Cmax

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
    
    n = 250
    for k in range(19):
        print "Class: ", k
        c = trainOneClass(Xtrain[0:n], ytrain[k][0:n])
        
        clf = svm.SVC(C = c, kernel = 'rbf', class_weight='auto')
        clf.fit(Xtrain[0:n], ytrain[k][0:n])

        y = clf.predict(Xtrain[n:])
        y1 = [y[i] for i in range(len(y))]
        
        fpr, tpr, thresholds = metrics.roc_curve(ytrain[k][n:], y1, pos_label=1)
        auc = metrics.auc(fpr,tpr)

        print "  C chosen:  ", c
        print "  AUC score: ", auc
        print "  Real train: ", ytrain[k][n:]
        print "  Pred train: ", y1
        
    
if __name__ == "__main__":
    main()