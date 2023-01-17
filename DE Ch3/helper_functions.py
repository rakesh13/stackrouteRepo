#to calculate a suite of test statistics of the performance of your model.
import sklearn.metrics as mt
import time 
import numpy as np
def printModelSummary(actual, predicted):
    '''
        Method to print out model summaries
    '''
    print('Overall accuracy of the model is {0:.2f} percent'.format((actual == predicted).sum()/ len(actual) * 100))
    print('Classification report: \n', mt.classification_report(actual, predicted))
    print('Confusion matrix: \n', mt.confusion_matrix(actual, predicted))
    print('Receiver Operating Characteristic (ROC): ', mt.roc_auc_score(actual, predicted))

#Calculate Execution time of a method    
def timeit(method):
    '''
        A decorator to time how long it takes to estimate
        the models
    '''

    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()

        print('The method {0} took {1:2.2f} sec to run.' \
              .format(method.__name__, end-start))
        return result

    return timed

#splitting data in to test and train
def split_data(data, y, x = 'All', test_size = 0.33):
    '''
        Method to split the data into training and testing
    '''
    import sys

    # dependent variable
    variables = {'y': y}

    # and all the independent
    if x == 'All':
        allColumns = list(data.columns)
        allColumns.remove(y)
        variables['x'] = allColumns
    else:
        if type(x) != list:
            print('The x parameter has to be a list...')
            sys.exit(1)
        else:
            variables['x'] = x

    # create a variable to flag the training sample
    data['train']  = np.random.rand(len(data)) < (1 - test_size) 

    # split the data into training and testing
    train_x = data[data.train] [variables['x']]
    train_y = data[data.train] [variables['y']]
    test_x  = data[~data.train][variables['x']]
    test_y  = data[~data.train][variables['y']]

    return train_x, train_y, test_x, test_y, variables['x']


def pseudo_F(X, labels, centroids):
    '''
        The pseudo F statistic :
        pseudo F = [( [(T - PG)/(G - 1)])/( [(PG)/(n - G)])] 
        The pseudo F statistic was suggested by 
        Calinski and Harabasz (1974) in 
        Calinski, T. and J. Harabasz. 1974. 
            A dendrite method for cluster analysis. 
            Commun. Stat. 3: 1-27.
            http://dx.doi.org/10.1080/03610927408827101

        We borrowed this code from 
        https://github.com/scampion/scikit-learn/blob/master/
        scikits/learn/cluster/__init__.py

        However, it had an error so we altered how B is
        calculated.
    '''
    center = np.mean(X,axis=0)
    u, count = np.unique(labels, return_counts=True)

    B = np.sum([count[i] * ((cluster - center)**2)
        for i, cluster in enumerate(centroids)])

    X = X.as_matrix()
    W = np.sum([(x - centroids[labels[i]])**2 
                 for i, x in enumerate(X)])

    k = len(centroids)
    n = len(X)

    return (B / (k-1)) / (W / (n-k))

def davis_bouldin(X, labels, centroids):
    '''
        The Davis-Bouldin statistic is an internal evaluation
        scheme for evaluating clustering algorithms. It
        encompasses the inter-cluster heterogeneity and 
        intra-cluster homogeneity in one metric.

        The measure was introduced by 
        Davis, D.L. and Bouldin, D.W. in 1979.
            A Cluster Separation Measure
            IEEE Transactions on Pattern Analysis and 
            Machine Intelligence, PAMI-1: 2, 224--227

            http://dx.doi.org/10.1109/TPAMI.1979.4766909
    '''
    distance = np.array([
        np.sqrt(np.sum((x - centroids[labels[i]])**2)) 
        for i, x in enumerate(X.as_matrix())])

    u, count = np.unique(labels, return_counts=True)

    Si = []

    for i, group in enumerate(u):
        Si.append(distance[labels == group].sum() / count[i])

    Mij = []

    for centroid in centroids:
        Mij.append([
            np.sqrt(np.sum((centroid - x)**2)) 
            for x in centroids])

    Rij = [] 
    for i in range(len(centroids)):
        Rij.append([
            0 if i == j 
            else (Si[i] + Si[j]) / Mij[i][j] 
            for j in range(len(centroids))])

    Di = [np.max(elem) for elem in Rij]

    return np.array(Di).sum() / len(centroids)

def printClustersSummary(data, labels, centroids):
    '''
        Helper method to automate models assessment
    '''
    print('Pseudo_F: ', pseudo_F(data, labels, centroids))
    print('Davis-Bouldin: ',davis_bouldin(data, labels, centroids))
    
def plot_components(z, y, color_marker, **f_params):
    
    #Produce and save the chart presenting 3 principalcomponents
    
    # import necessary modules
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # convert the dependent into a Numpy array
    # this is done so z and y are in the same format
    y_np = y

    # do it only, however, if y is not NumPy array
    if type(y_np) != np.array:
        y_np = np.array(y_np)

    # create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the dots
    for i, j in enumerate(np.unique(y_np)):
        ax.scatter(
            z[y_np == j, 0], 
            z[y_np == j, 1], 
            z[y_np == j, 2], 
            c=color_marker[i][0], 
            marker=color_marker[i][1])

    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')
    ax.set_zlabel('Third component')

    # save the figure
    plt.savefig(**f_params)
    
    
    
    
    