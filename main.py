import random
import warnings
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def read_labels(file):
    flowers_file = open(file, 'r')

    labels = []

    Lines = flowers_file.readlines()

    for line in Lines:
        labels.append(int(line.strip('\n')))

    return labels


def get_images_names():
    images = glob('jpg\*')
    seg_images = glob('flower_sementations\*')
    # seg_images = glob('jpg\*')

    return images, seg_images


def rescale(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def blur(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return blur


def rgb_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def hsv_img(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vectorized = hsv_img.reshape((-1, 3))
    vectorized = np.float32(vectorized)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    result_image = res.reshape(img.shape)
    result_image = rescale(img,32,64)
    result_image = rgb_to_gray(result_image)

    return result_image


def hog_g(img):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                        multichannel=True)
    hog_image = rescale(hog_image,32,64)

    return hog_image


def orb_boundary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_gray = cv2.Canny(gray, 30, 200)

    orb = cv2.ORB_create()
    keypoints = orb.detect(edge_gray, None)
    keypoints, descriptors = orb.compute(edge_gray, keypoints)
    kp_img = cv2.drawKeypoints(edge_gray, keypoints, None, color=(0, 255, 0), flags=0)

    kp_img = rgb_to_gray(kp_img)
    kp_img = rescale(kp_img,32,64)
    return kp_img


def orb_foreground(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    keypoints, descriptors = orb.compute(gray, keypoints)

    kp_img = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), flags=0)

    kp_img = rgb_to_gray(kp_img)
    kp_img = rescale(kp_img,32,64)
    return kp_img


def imgWlabels(label_list, labels):
    print("Images that you want to classify")
    flag = False
    temp = []

    for index, label in enumerate(labels):
        img = plt.imread(images[index])
        if label in label_list and label not in temp:
            plt.title(f"{label}.LABEL")
            plt.imshow(img)
            plt.show()
            temp.append(label)


def feature_extract(label_list, labels, get_hsv_kmeans=False, get_hog=False, get_orb_b=False, get_orb_frg=False):
    hsv_lb = []
    flatten = []

    hsv_label.clear()

    for index, label in enumerate(labels):
        img = plt.imread(seg_images[index])
        if label in (label_list):
            resc = rescale(img, 128, 256)

            if get_hsv_kmeans:
                hs = hsv_img(resc)
                flat_hs = np.array(hs).ravel()
                flatten = np.hstack((flatten, flat_hs))

            if get_hog:
                hogged = hog_g(resc)
                flag_hog = np.array(hogged).ravel()
                flatten = np.hstack((flatten, flag_hog))

            if get_orb_b:
                orb_b = orb_boundary(resc)
                flat_orb_b = np.array(orb_b).ravel()
                flatten = np.hstack((flatten, flat_orb_b))

            if get_orb_frg:
                orb_frg = orb_foreground(resc)
                flat_orb_frg = np.array(orb_frg).ravel()
                flatten = np.hstack((flatten, flat_orb_frg))

            if hsv_label.get(label) is not None:
                hsv_lb = hsv_label.get(label)
                hsv_lb = np.vstack((hsv_lb, flatten))
                hsv_label[label] = hsv_lb

            else:
                hsv_label[label] = flatten

            flatten = []


def do_pca(data, plot=False):
    X = data

    pca = PCA()
    scores = pca.fit_transform(X)

    if plot:
        plt.plot(pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
        plt.title('Explained Variance by Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid()
        plt.show()

    n = int(input("Please enter n_components: "))
    pca = PCA(n_components=n)
    pca.fit(X)

    X = pca.transform(X)
    return X


def optimize_KNN(X_train, y_train):
    params = {'n_neighbors': [5, 7, 9, 11, 13, 15],
              'weights': ['uniform', 'distance'],
              'metric': ['minkowski', 'euclidean', 'manhattan']}

    knn_gs = GridSearchCV(KNeighborsClassifier(), params, verbose=1, cv=5, n_jobs=-1)
    knn_gs_res = knn_gs.fit(X_train, y_train)
    best_params = knn_gs_res.best_params_

    print("Best score: ", knn_gs_res.best_score_)
    print("Best parameters: ", best_params)

    return best_params


def KNN(data, label):
    print("********KNN Classifier*************")
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

    best_params = optimize_KNN(X_train, y_train)

    knn = KNeighborsClassifier(**best_params)
    knn.fit(X_train, y_train)

    y_hat = knn.predict(X_train)
    y_pred = knn.predict(X_test)

    print("\nConfusion Matrix is,\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print('Training set accuracy: ', metrics.accuracy_score(y_train, y_hat))
    print('Test set accuracy: ', metrics.accuracy_score(y_test, y_pred))


def optimize_randomForest(X_train, y_train):
    params = {"max_features": ['auto', 'sqrt'], "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators": [100, 300], "criterion": ["gini"]}

    rf = RandomForestClassifier()
    rf_cv = GridSearchCV(rf, params, cv=10)
    rf_cv.fit(X_train, y_train)

    best_params = rf_cv.best_params_

    print("Tuned hpyerparameters :(best parameters) ", rf_cv.best_params_)
    print("Accuracy :", rf_cv.best_score_)

    return best_params


def randomForest(data, label):
    print("********Random Forest*************")
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

    best_params = optimize_randomForest(X_train, y_train)

    rf = RandomForestClassifier(**best_params)
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)

    print("\nConfusion Matrix is,\n", confusion_matrix(y_test, y_pred_test))
    print("\nClassification report:\n", classification_report(y_test, y_pred_test))
    print('Test set accuracy: ', metrics.accuracy_score(y_test, y_pred_test))


def optimize_logisticRegression(X_train, y_train):
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              "penalty": ["l1", "l2"]}

    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, params, cv=10)
    logreg_cv.fit(X_train, y_train)

    best_params = logreg_cv.best_params_

    print("Tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print("Accuracy :", logreg_cv.best_score_)

    return best_params


def logisticRegression(data, label):
    print("********Logistic Regression*************")
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

    best_params = optimize_logisticRegression(X_train, y_train)

    logreg = LogisticRegression(**best_params)
    logreg.fit(X_train, y_train)
    y_pred_test = logreg.predict(X_test)

    print("\nConfusion Matrix is,\n", confusion_matrix(y_test, y_pred_test))
    print("\nClassification report:\n", classification_report(y_test, y_pred_test))
    print('Test set accuracy: ', metrics.accuracy_score(y_test, y_pred_test))


def optimize_decisionTree(X_train, y_train):
    params_dt = {
        "criterion": ("gini", "entropy"),
        "splitter": ("best", "random"),
        "max_depth": (list(range(1, 20))),
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": list(range(1, 20)),
    }

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_cv = GridSearchCV(tree_clf, params_dt, scoring="accuracy", n_jobs=-1, cv=5)
    tree_cv.fit(X_train, y_train)
    best_params = tree_cv.best_params_
    best_score = tree_cv.best_score_

    print(f"Best score: {best_score}")
    print(f"Best paramters: {best_params}")

    return best_params


def decisionTree(data, label):
    print("********DecisionTree*************")
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

    best_params = optimize_decisionTree(X_train, y_train)

    tree_clf = DecisionTreeClassifier(**best_params)
    tree_clf.fit(X_train, y_train)
    y_pred_test = tree_clf.predict(X_test)

    print("\nConfusion Matrix is,\n", confusion_matrix(y_test, y_pred_test))
    print("\nClassification report:\n", classification_report(y_test, y_pred_test))
    print("Accuracy on test set with tree: {:.2f}".format(metrics.accuracy_score(y_test, y_pred_test)))


def optimize_SVM(X_train, y_train, param="OVO"):
    if param == "OVO":
        SVM_OVO = OneVsOneClassifier(SVC(decision_function_shape='ovo'))

        param_grid = {'estimator__C': [0.1, 1, 10, 50, 100],
                      'estimator__gamma': [0.1, 0.01],
                      'estimator__kernel': ['rbf', 'poly', 'sigmoid']}

        model_svm = GridSearchCV(SVM_OVO, param_grid, cv=5, scoring='accuracy')
        model_svm.fit(X_train, y_train)

        best_params = model_svm.best_params_
        best_score = model_svm.best_score_

        best_C = best_params.get('estimator__C')
        best_gamma = best_params.get('estimator__gamma')
        best_kernel = best_params.get('estimator__kernel')

        print(f"Best score: {best_score}")
        print(f"Best parameters: {best_params}")

    elif param == "OVA":

        SVM_OVA = OneVsRestClassifier(SVC(decision_function_shape='ovr'))
        param_grid = {'estimator__C': [10, 100],
                      'estimator__gamma': [0.1, 0.01],
                      'estimator__kernel': ['linear', 'poly', 'sigmoid', 'rbf']}
        model_svm = GridSearchCV(SVM_OVA, param_grid, cv=4, scoring='accuracy')
        model_svm.fit(X_train, y_train)

        best_params = model_svm.best_params_
        best_score = model_svm.best_score_

        print(f"Best score: {best_score}")
        print(f"Best paramters: {best_params}")

        best_C = best_params.get('estimator__C')
        best_gamma = best_params.get('estimator__gamma')
        best_kernel = best_params.get('estimator__kernel')

    return best_C, best_gamma, best_kernel


def SVM(data, label, param="OVO"):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

    if param == "OVO":
        print("**********OVO SVM************")
        best_C, best_gamma, best_kernel = optimize_SVM(X_train, y_train, param)

        SVM_OVO = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma, decision_function_shape='ovo')
        SVM_OVO.fit(X_train, y_train)

        y_predict_OVO = SVM_OVO.predict(X_test)

        print("\nClassification report:\n", classification_report(y_test, y_predict_OVO))
        print("Accuracy: ", metrics.accuracy_score(y_test, y_predict_OVO))

    elif param == "OVA":
        print("*************OVA SVM***********")
        best_C, best_gamma, best_kernel = optimize_SVM(X_train, y_train, param)
        SVM_OVA = OneVsRestClassifier(estimator=SVC(C=best_C, gamma=best_gamma,
                                                    kernel=best_kernel, decision_function_shape='ovr'))
        SVM_OVA.fit(X_train, y_train)

        y_predict_OVA = SVM_OVA.predict(X_test)

        print("\nClassification report:\n", classification_report(y_test, y_predict_OVA))
        print("Accuracy: ", metrics.accuracy_score(y_test, y_predict_OVA))


def find_best_f(label_list, labels):
    subset = [[True, False, False, False], [False, True, False, False], [False, False, True, False],
              [False, False, False, True],
              [True, True, False, False], [True, False, True, False], [True, False, False, True],
              [False, True, True, False], [False, True, False, True],
              [False, False, True, True], [True, True, True, False], [True, True, False, True],
              [True, False, True, True], [False, True, True, True],
              [True, True, True, True]]

    print(f"CLASSIFY WITH {label_list} LABELS")

    for i in range(15):
        print("*******************SUBSET*****************", subset[i][0], subset[i][1], subset[i][2], subset[i][3])
        feature_extract(label_list, labels, subset[i][0], subset[i][1], subset[i][2], subset[i][3])
        y = []
        X = []

        for i in range(len(label_list)):
            if len(X) == 0:
                X = np.vstack(hsv_label.get(label_list[i]))
            else:
                X = np.vstack((X, hsv_label.get(label_list[i])))

            for k in range(len(hsv_label.get(label_list[i]))):
                y.append(label_list[i])

        print("Before PCA ", X.shape)

        y = np.asarray(y)

        classify(X, y)


def classify_w_size(labels):
    size = [2, 5, 15, 50, 102]

    for i in size:
        label_list = random.sample(range(1, 103), i)
        find_best_f(label_list, labels)


def classify(label_list, labels):
    X, y = feature_extract_with_list(label_list, labels)

    randomForest(X, y)


def feature_extract_with_list(label_list, labels, feature_list=[True, True, True, True]):
    feature_extract(label_list, labels, feature_list[0], feature_list[1], feature_list[2], feature_list[3])
    y = []
    X = []

    for i in range(len(label_list)):
        if len(X) == 0:
            X = np.vstack(hsv_label.get(label_list[i]))
        else:
            X = np.vstack((X, hsv_label.get(label_list[i])))

        for k in range(len(hsv_label.get(label_list[i]))):
            y.append(label_list[i])

    print("Before PCA ", X.shape)

    y = np.asarray(y)

    # print("Labels for y",y,y.shape)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X = do_pca(X, True)
    print("After PCA", X.shape)

    return X, y


def best_features_classify(label_list, labels, classifier="KNN", feature_list=[True, True, True, True]):
    print(f"CLASSIFY WITH {label_list} LABELS")
    X, y = feature_extract_with_list(label_list, labels, feature_list)

    if classifier == "KNN":
        KNN(X, y)
    elif classifier == "dt":
        decisionTree(X, y)
    elif classifier == "log":
        logisticRegression(X, y)
    elif classifier == "svm_ovo":
        SVM(X, y, "OVO")
    else:
        SVM(X, y, "OVA")


def bagging(label_list, labels, classifier="KNN", feature_list=[True, True, True, True]):
    X, y = feature_extract_with_list(label_list, labels, feature_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    if classifier == "KNN":
        print("********Bagging KNN*************")
        clf = BaggingClassifier(
            base_estimator=KNeighborsClassifier(metric="minkowski", n_neighbors=5, weights="distance"), n_estimators=10,
            random_state=42).fit(X_train, y_train)
    elif classifier == "dt":
        print("********Bagging Decision Tree*************")
        clf = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=19, splitter="best"),
            n_estimators=10, random_state=42).fit(X_train, y_train)
    elif classifier == "log":
        print("********Bagging Logistic Regression*************")
        clf = BaggingClassifier(base_estimator=LogisticRegression(C=0.001, penalty='l2'), n_estimators=10,
                                random_state=42).fit(X_train, y_train)
    elif classifier == "svm_ovo":
        print("********Bagging SVM OVO*************")
        clf = BaggingClassifier(base_estimator=SVC(kernel="poly", C=0.1, gamma=0.1, decision_function_shape='ovo'),
                                n_estimators=10, random_state=42).fit(X_train, y_train)
    else:
        print("********Bagging SVM OVA*************")
        clf = BaggingClassifier(
            base_estimator=OneVsRestClassifier(SVC(kernel="linear", C=10, gamma=0.1, decision_function_shape='ovr')),
            n_estimators=10, random_state=42).fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    print("\nConfusion Matrix is,\n", confusion_matrix(y_test, y_predict))
    print("\nClassification report:\n", classification_report(y_test, y_predict))
    print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))


if __name__ == "__main__":
    labels = read_labels("flowers_labels.txt")
    images, seg_images = get_images_names()

    hsv_label = dict()

    # 2 Different Classes
    label_list = [37, 20]
    imgWlabels(label_list, labels)

    # KNN
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="KNN", feature_list=feature_list)

    # DecisionTree
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="dt", feature_list=feature_list)

    # Logistic
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="log", feature_list=feature_list)

    # SVM-OVA
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="svm_ova", feature_list=feature_list)

    # Class SVM-OVO
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="svm_ovo", feature_list=feature_list)
    #
    # 2 Similar Classes
    label_list = [69, 64]
    imgWlabels(label_list, labels)

    # KNN
    feature_list = [False, True, False, False]
    best_features_classify(label_list, labels, classifier="KNN", feature_list=feature_list)

    # DecisionTree
    feature_list = [False, True, False, False]
    best_features_classify(label_list, labels, classifier="dt", feature_list=feature_list)

    # Logistic
    feature_list = [False, True, False, False]
    best_features_classify(label_list, labels, classifier="log", feature_list=feature_list)

    # SVM-OVA
    feature_list = [False, True, False, False]
    best_features_classify(label_list, labels, classifier="svm_ova", feature_list=feature_list)

    # SVM-OVO
    feature_list = [False, True, False, False]
    best_features_classify(label_list, labels, classifier="svm_ovo", feature_list=feature_list)

    ##########################################################################################################
    #
    # 10 Similar Classes
    label_list = [75, 81, 69, 64, 70, 49, 7, 67, 2, 83]
    imgWlabels(label_list, labels)

    # KNN
    feature_list = [True, True, False, True]
    best_features_classify(label_list, labels, classifier="KNN", feature_list=feature_list)

    # DecisionTree
    feature_list = [True, False, False, True]
    best_features_classify(label_list, labels, classifier="dt", feature_list=feature_list)

    # Logistic
    feature_list = [True, True, False, True]
    best_features_classify(label_list, labels, classifier="log", feature_list=feature_list)

    # SVM-OVA
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="svm_ova", feature_list=feature_list)

    # SVM-OVO
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="svm_ovo", feature_list=feature_list)

    # 10 Different Classes
    label_list = [77, 73, 88, 89, 81, 84, 29, 82, 41, 8]
    imgWlabels(label_list, labels)

    # KNN
    feature_list = [True, True, False, True]
    best_features_classify(label_list, labels, classifier="KNN", feature_list=feature_list)

    # DecisionTree
    feature_list = [True, False, False, True]
    best_features_classify(label_list, labels, classifier="dt", feature_list=feature_list)

    # Logistic
    feature_list = [True, True, False, True]
    best_features_classify(label_list, labels, classifier="log", feature_list=feature_list)

    # SVM-OVA
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="svm_ova", feature_list=feature_list)

    # SVM-OVO
    feature_list = [True, True, False, False]
    best_features_classify(label_list, labels, classifier="svm_ovo", feature_list=feature_list)
    #
    ###########################################################################################################

    # Random Forest
    # 10 Different Classes
    label_list = [77, 73, 88, 89, 81, 84, 29, 82, 41, 8]
    classify(label_list, labels)

    ###########################################################################################################

    #####Ensemble Learning

    # 10 Different Classes
    label_list = [77, 73, 88, 89, 81, 84, 29, 82, 41, 8]

    # Voting - you can change label list for try different number of classes
    clf1 = LogisticRegression(C=0.001, penalty='l2')
    clf2 = DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=19, splitter="best")
    clf3 = KNeighborsClassifier(metric="minkowski", n_neighbors=5, weights="distance")
    clf4 = SVC(kernel="poly", C=0.1, gamma=0.1, decision_function_shape='ovo')
    clf5 = OneVsRestClassifier(SVC(kernel="linear", C=10, gamma=0.1, decision_function_shape='ovr'))

    eclf1 = VotingClassifier(
        estimators=[('lr', clf1), ('dt', clf2), ('knn', clf3), ('svm_ovo', clf4), ('svm_ova', clf5)], voting='hard')

    X, y = feature_extract_with_list(label_list, labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    eclf1 = eclf1.fit(X_train, y_train)
    predict = eclf1.predict(X_test)
    print("********Ensemble Learning Voting*************")
    print("\nConfusion Matrix is,\n", confusion_matrix(y_test, predict))
    print("\nClassification report:\n", classification_report(y_test, predict))
    print("Accuracy: ", metrics.accuracy_score(y_test, predict))

    # Bagging

    feature_list = [True, True, False, True]
    bagging(label_list, labels, classifier="KNN", feature_list=feature_list)

    feature_list = [True, True, False, True]
    bagging(label_list, labels, classifier="log", feature_list=feature_list)

    feature_list = [True, False, False, True]
    bagging(label_list, labels, classifier="dt", feature_list=feature_list)

    feature_list = [True, True, False, False]
    bagging(label_list, labels, classifier="svm_ovo", feature_list=feature_list)

    feature_list = [True, True, False, False]
    bagging(label_list, labels, classifier="svm_ova", feature_list=feature_list)

    #########################################################################################################
    # 102 Class Testing for According to our decision for Best Classifier
    label_list = range(1,103)

    # Logistic
    feature_list = [True, True, False, True]
    X, y = feature_extract_with_list(label_list, labels, feature_list=feature_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    log = LogisticRegression(C=0.001, penalty='l2')
    log.fit(X_train, y_train)
    y_pred_test = log.predict(X_test)

    print("\nConfusion Matrix is,\n", confusion_matrix(y_test, y_pred_test))
    print("\nClassification report:\n", classification_report(y_test, y_pred_test))
    print('Test set accuracy: ', metrics.accuracy_score(y_test, y_pred_test))
    #
    # Bagging
    # Logistic
    # feature_list = [True, True, False, True]
    # bagging(label_list, labels, classifier="log", feature_list=feature_list)

    # Random Forest
    X, y = feature_extract_with_list(label_list, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rf = RandomForestClassifier(bootstrap=False, criterion="gini", max_features="auto", min_samples_leaf=1,
                                min_samples_split=3, n_estimators=300)

    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)

    print("\nConfusion Matrix is,\n", confusion_matrix(y_test, y_pred_test))
    print("\nClassification report:\n", classification_report(y_test, y_pred_test))
    print('Test set accuracy: ', metrics.accuracy_score(y_test, y_pred_test))
