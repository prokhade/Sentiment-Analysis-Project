import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

def usage():
    print("Usage:")
    print("python %s C:/Users/Prateek/Desktop/IUB/SA/Project/data/review_polarity/txt_sentoken" % sys.argv[0])


# if len(sys.argv) < 2:
#     usage()
#     print(sys.argv[0])
#     print(sys.argv[1])
#     sys.exit(1)

data_dir = "C:/Users/Prateek/Desktop/IUB/SA/Project/data/review_polarity/txt_sentoken/"
classes = ['pos', 'neg']

# Read the data
train_data = []
train_labels = []
test_data = []
test_labels = []
for curr_class in classes:
    dirname = os.path.join(data_dir, curr_class)
    for fname in os.listdir(dirname):
        with open(os.path.join(dirname, fname), 'r') as f:
            content = f.read()
            if fname.startswith('cv9'):
                test_data.append(content)
                test_labels.append(curr_class)
            else:
                train_data.append(content)
                train_labels.append(curr_class)

# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# # Perform classification with SVM, kernel=rbf
# classifier_rbf = svm.SVC()
# t0 = time.time()
# classifier_rbf.fit(train_vectors, train_labels)
# t1 = time.time()
# prediction_rbf = classifier_rbf.predict(test_vectors)
# t2 = time.time()
# time_rbf_train = t1 - t0
# time_rbf_predict = t2 - t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1 - t0
time_linear_predict = t2 - t1

# Perform classification with SVM, kernel=poly, degree=2
classifier_poly = svm.SVC(kernel='poly', degree=2)
t0 = time.time()
classifier_poly.fit(train_vectors, train_labels)
t1 = time.time()
prediction_poly = classifier_poly.predict(test_vectors)
t2 = time.time()
time_poly_train = t1 - t0
time_poly_predict = t2 - t1

# # Print results in a nice table
# print("Results for SVC(kernel=rbf)")
# print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
# print(classification_report(test_labels, prediction_rbf))
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_labels, prediction_linear))
print("Results for SVC(kernel=poly)")
print("Training time: %fs; Prediction time: %fs" % (time_poly_train, time_poly_predict))
print(classification_report(test_labels, prediction_poly))
