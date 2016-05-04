import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def usage():
    print("Usage:")
    print("python %s C:/Users/Aditya/Desktop/IUB/SA/Project/data/review_polarity/txt_sentoken" % sys.argv[0])


# if len(sys.argv) < 2:
#     usage()
#     print(sys.argv[0])
#     print(sys.argv[1])
#     sys.exit(1)

data_dir = "C:/Users/Aditya/Desktop/IUB/SA/Project/data/review_polarity/txt_sentoken/"
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

# Perform classification with SVM, kernel=rbf
classifier_knn_5 = KNeighborsClassifier(n_neighbors=5)
t0 = time.time()
classifier_knn_5.fit(train_vectors, train_labels)
t1 = time.time()
prediction_knn_5 = classifier_knn_5.predict(test_vectors)
t2 = time.time()
time_knn_5_train = t1 - t0
time_knn_5_predict = t2 - t1


# Perform classification with SVM, kernel=rbf
classifier_knn_50 = KNeighborsClassifier(n_neighbors=50)
t0 = time.time()
classifier_knn_50.fit(train_vectors, train_labels)
t1 = time.time()
prediction_knn_50 = classifier_knn_50.predict(test_vectors)
t2 = time.time()
time_knn_50_train = t1 - t0
time_knn_50_predict = t2 - t1


# Print results in a nice table
print("Results for knn (k=5")
print("Training time: %fs; Prediction time: %fs" % (time_knn_5_train, time_knn_5_predict))
print(classification_report(test_labels, prediction_knn_5))

print("Results for knn (k=50")
print("Training time: %fs; Prediction time: %fs" % (time_knn_50_train, time_knn_50_predict))
print(classification_report(test_labels, prediction_knn_50))
