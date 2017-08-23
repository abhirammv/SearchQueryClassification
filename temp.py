import pandas as pandas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
import numpy as numpy
from sklearn.cross_validation import cross_val_score
import os as os


def set_working_directory_for_training_data(location):
    pwd = os.getcwd()
    os.chdir(os.path.dirname(location))
    training_data = pandas.read_csv(os.path.basename(location))
    os.chdir(pwd)
    return training_data


def analyze_training_data(training_data):
    training_data.isnull().values.any()
    training_data.describe()
    training_data.corr()


def normalize_training_data(data, start_col, end_col):
    standard_scalar = StandardScaler()
    data[:, start_col:end_col] = standard_scalar.fit_transform(data[:, start_col:end_col])
    return data


def filter_predictors_from_data_set(training_data, start, end):
    return training_data.iloc[:, start:  end].values


def filter_output_var_from_data_set(training_data, index):
    return training_data.iloc[:, index].values


def perform_cross_validation(model, predictors, output_var, cv_fold):
    cross_validation = cross_val_score(model, predictors, output_var, scoring="accuracy", cv=cv_fold)
    if type(model).__name__ != "RFE":
        print(type(model).__name__ + " --> " + str(cross_validation.mean()))


def perform_deletion_from_predictor_set(predictors, index):
    modified_array = numpy.array(predictors)
    modified_array = numpy.delete(modified_array, index, axis=1)
    return modified_array


def get_training_data():
    training_data_set_location = "C:/Deepankit/CourseWork/DM/project/data_set/training.csv"
    training_data = set_working_directory_for_training_data(training_data_set_location)
    return training_data


def build_classification_models_on_one_less_predictor(predictors, output):
    for index in range(0, len(predictors[0])):
        print("Performing model classification by removing predictor at column index {}".format(index))
        pruned_predictors = perform_deletion_from_predictor_set(predictors, index)
        build_classification_models(pruned_predictors, output)


def build_classification_models(predictors, output):
    # logistic regression
    # model = LogisticRegression()
    # perform_cross_validation(model, predictors, output, 10)

    # KNN with Auto algorithm
    # model = KNeighborsClassifier(n_neighbors=index + 1, weights='uniform', algorithm='auto')
    # perform_cross_validation(model, predictors, output, 10)

    # KNN with KD Tree algorithm
    #    model = KNeighborsClassifier(n_neighbors=index + 1, weights='uniform', algorithm='kd_tree')
    #   perform_cross_validation(model, predictors, output, 10)

    # Ada boost
    # model = AdaBoostClassifier(n_estimators=100)
    # perform_cross_validation(model, predictors, output, 10)

    # GradientBoostingClassifier with 50 estimators
    # model = GradientBoostingClassifier(n_estimators=50 * (index + 1), loss="exponential")
    # perform_cross_validation(model, predictors, output, 10)
    # GradientBoostingClassifier with 100 estimators
    # model = GradientBoostingClassifier(n_estimators=100, loss="exponential")
    # perform_cross_validation(model, predictors, output, 10)
    # GradientBoostingClassifier with 200 estimators
    # model = GradientBoostingClassifier(n_estimators=200, loss="exponential")
    # perform_cross_validation(model, predictors, output, 10)
    # GradientBoostingClassifier with 300 estimators
    # model = GradientBoostingClassifier(n_estimators=300, loss="exponential")
    # perform_cross_validation(model, predictors, output, 10)

    # Random Forest
    for index in range(10):
        model = RandomForestClassifier(n_estimators=50 * (index + 1), min_samples_split=2, max_features='auto',
                                       bootstrap=True)
        perform_cross_validation(model, predictors, output, 10)

    # SVM with RBF
    # model = svm.SVC(kernel='rbf')
    # perform_cross_validation(model, predictors, output, 10)
    # SVM with Gaussian
    # model = svm.SVC(kernel='linear')
    # perform_cross_validation(model, predictors, output, 10)

    # Naive Bayes with GaussianNB
    # model = GaussianNB()
    # perform_cross_validation(model, predictors, output, 10)

    # Decision Tree with max depth 5
    # model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5)
    # perform_cross_validation(model, predictors, output, 10)
    # Decision Tree with max depth 10
    # model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10)
    # perform_cross_validation(model, predictors, output, 10)
    # Decision Tree with max depth 15
    # model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=15)
    # perform_cross_validation(model, predictors, output, 10)
    # Decision Tree with max depth 20
    # model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20)
    # perform_cross_validation(model, predictors, output, 10)

    # Bagging with Bootstrap
    # model = BaggingClassifier(bootstrap=True)
    # perform_cross_validation(model, predictors, output, 10)

    # Bagging without Bootstrap
    # model = BaggingClassifier(bootstrap=False)
    # perform_cross_validation(model, predictors, output, 10)


def perform_feature_selection(feature_index, predictors, output):
    # logistic regression
    model = LogisticRegression()
    fit_rfe(model, feature_index, predictors, output)

    # Ada boost
    model = AdaBoostClassifier()
    fit_rfe(model, feature_index, predictors, output)

    # GradientBoostingClassifier with 300 estimators
    model = GradientBoostingClassifier(n_estimators=300)
    fit_rfe(model, feature_index, predictors, output)

    # Decision Tree with max depth 20
    model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20)
    fit_rfe(model, feature_index, predictors, output)


def fit_rfe(model, feature_index, predictors, output):
    print(type(model).__name__)
    model = RFE(model, feature_index)
    perform_cross_validation(model, predictors, output, 10)
    model = model.fit(predictors, output)
    print(model.support_)
    print(model.ranking_)


def retrieve_features_based_on_index(predictors, start, end):
    if start == end:
        return predictors[:, start]
    return predictors[:, start: end + 1]


def build_classification(predictors, output):
    classification_predictors = predictors
    for data_index in range(len(classification_predictors[0])):
        classification_predictors = predictors
        print("Performing RFE for total features #: " + str(data_index + 1))
        classification_predictors = retrieve_features_based_on_index(classification_predictors, 0, data_index)
        if data_index == 0:
            classification_predictors = numpy.reshape(classification_predictors, (-1, 1))
        perform_feature_selection(data_index + 1, classification_predictors, output)


def enhance_predictors_by_transformation(predictors, start, end):
    predictors[:, start:end] = numpy.log(1 + predictors[:, start: end])
    return predictors


def add_query_weight_to_predictor_set(predictors):
    new_column = numpy.zeros((predictors.shape[0], 1))
    predictors = numpy.column_stack((predictors, new_column))
    column = predictors.shape[1] - 1
    for index in range(0, predictors.shape[0]):
        if predictors[index, 0] == 1:
            predictors[index, column] = numpy.log(10.00)
        elif predictors[index, 0] == 2:
            predictors[index, column] = 15.50
        elif predictors[index, 0] == 3:
            predictors[index, column] = 9.00
        elif predictors[index, 0] == 4:
            predictors[index, column] = 4.35
        elif predictors[index, 0] == 5:
            predictors[index, column] = 3.20
        elif predictors[index, 0] == 6:
            predictors[index, column] = 0.84
        elif predictors[index, 0] == 7:
            predictors[index, column] = 0.42
        else:
            pass
    return predictors


def add_query_sum_to_training_data_set(training_data):
    new_column = numpy.zeros((training_data.shape[0], 1))

    query_id_column = training_data['query_id']
    url_id_column = training_data['url_id']
    new_column[0] = 1
    for index in range(1, training_data.shape[0]):
        if (query_id_column[index] == query_id_column[index - 1] and url_id_column[index] == url_id_column[
                index - 1] + 1):
            new_column[index] = new_column[index - 1] + 1
        else:
            new_column[index] = 1
    return new_column


if __name__ == '__main__':
    training_data_set = get_training_data()
    query_sum = add_query_sum_to_training_data_set(training_data_set)
    predictors_set = filter_predictors_from_data_set(training_data_set, 2, 12)
    predictors_set = numpy.column_stack((predictors_set, query_sum))
    predictors_set = add_query_weight_to_predictor_set(predictors_set)
    predictors_set = normalize_training_data(predictors_set, 4, 8)
    predictors_set = enhance_predictors_by_transformation(predictors_set, 2, 8)
    start_index = 0
    end_index = 13
    predictors_set = retrieve_features_based_on_index(predictors_set, start_index, end_index)
    output_set = filter_output_var_from_data_set(training_data_set, 12)

    # build classification model
    # build_classification(predictors_set, output_set)

    print("Performing general classification model on all the features")
    build_classification_models(predictors_set, output_set)