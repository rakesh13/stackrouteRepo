#Classifying with Naïve Bayes
import helper_functions as hlp
import pandas as pd
import sklearn.naive_bayes as nb

@hlp.timeit
def fitNaiveBayes(data):

    #Build the Naive Bayes classifier
    
    # create the classifier object
    naiveBayes_classifier = nb.GaussianNB()

    # fit the model
    return naiveBayes_classifier.fit(data[0], data[1])

#reading data and storing to a dataframe
Read_csv_filename = 'bank_contacts.csv'
Read_csv_data = pd.read_csv(Read_csv_filename)

# split the data into training and testing
train_x, train_y, test_x,  test_y, labels = hlp.split_data(Read_csv_data, y = 'credit_application')

# train the model
classifier = fitNaiveBayes((train_x, train_y))

# classify the unseen data
predicted = classifier.predict(test_x)

# print out the results
hlp.printModelSummary(test_y, predicted)

print("Naive bayes Model fitted successfully")
