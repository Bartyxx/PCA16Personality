import tensorflow
import keras
from keras.utils import to_categorical
from keras.layers import Dense
import pandas as pd
import numpy as np
from columns import columns, drop_columns, non_drop_columns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from function import count_unique

"""Dataset URL"""
dataset_url = 'https://www.kaggle.com/datasets/anshulmehtakaggl/60k-responses-of-16-personalities-test-mbt'
"""16 personality test"""
URL = 'https://www.16personalities.com/it/test-della-personalita-gratis'


"""
DATASET EXPLNATION
LABEL EXPLENATION.
(Last columns, 'Personality'.
Every label is one personality, the label is composed by 4 letters:
----------------------------------> 1st letter: E Extrovert, I Introvert,
----------------------------------> 2nd letter: S Observant, N Intuitive,
----------------------------------> 3th letter: T Thinking, I Intuitive,
----------------------------------> 4th letter: J Judgment, P Perception.

The combination of 4 of this 8 types gives your personality.

FEATURE EXPLENATION.
(The first columns is a progressive ID, I have remove this column at the beginning of the project).
In the test you answer 100 question, in this dataset the question considered are only 60, one question correspond to one
column. The possible answer to every question are:
----------------------------------> -3 = Strongly Disagree
----------------------------------> -2 = Disagree
----------------------------------> -1 = Slightly Disagree
---------------------------------->  0 = Neutral
---------------------------------->  1 = Slightly Agree
---------------------------------->  2 = Agree 
---------------------------------->  3 = Strongly Agree
All the columns contain every value, they didn't have any missing values.
"""

if __name__ == "__main__":
    '''
    ---------------------------------------------------------------------------
                      TASK 1: Import and Create the datasets.
    ---------------------------------------------------------------------------
    Creation of a Pandas datasets from the 
    Kaggle CSV file
    ---------------------------------------------------------------------------
                                                                            '''
    ds = pd.read_csv('./data/16p.csv', sep=',', encoding='cp1252', low_memory=False)


    '''
    ---------------------------------------------------------------------------
                              TASK 2: Divide y and x.
    ---------------------------------------------------------------------------
    Remove the columns Personality(y) and Response 
    ID(useless for our purpose from the features. Assign Personality columns 
    to the variable y.
    ---------------------------------------------------------------------------   
                                                                            '''
    y = ds['Personality'].values
    x = ds.drop(columns=['Personality', 'Response Id']).values


    '''
    ---------------------------------------------------------------------------
                       TASK 3: Normalization of the labels
    ---------------------------------------------------------------------------
    Find the unique value on the y columns, then with the for cycle change 
    every value with a progressive numerical value.
    ---------------------------------------------------------------------------
                                                                            '''
    print('classes & distribution: ', np.unique(y, return_counts=True),'\n')
    label_names = np.unique(y)
    for i, label in enumerate(label_names):
        mask = y == label
        y[mask] = i
    y = y.astype(int)


    '''
    ---------------------------------------------------------------------------
                               TASK 4: x and y split.
    ---------------------------------------------------------------------------
    Divide the features and the labels with train_test_split:
                                                             test_size  -> 30%
                                                             train_size -> 70%
    ---------------------------------------------------------------------------
                                                                            '''
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3)

    '''
    ---------------------------------------------------------------------------
                                   TASK 5:  KNN
    ---------------------------------------------------------------------------
    Calculate the KNN.
    ---------------------------------------------------------------------------     
                                                                            '''
    KNN = KNeighborsClassifier(n_neighbors=7)
    KNN.fit(x_train, y_train)
    KNN_y_pred = KNN.predict(x_test)
    accuracyKNN = np.mean(y_test == KNN_y_pred)
    print(f'--> KNN Accuracy: {100 * accuracyKNN: .2f}% \n')

    modelli = ["KNN k = 3", "KNN k = 5", "KNN k = 7", "Linear SVM", "Non Linear SVM", "Neural Network"]
    results_mean = [98.80, 98.93, 98.94, 94.71, 98.51, 91 ]
    
    fig = plt.figure()
    fig.patch.set_facecolor('paleturquoise')  
    plt.bar(modelli, results_mean, 0.5, color='indigo', align='center', )
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on ten attempts")
    ax = plt.gca()
    ax.set_facecolor('lightskyblue')
    plt.show()

    '''
    ---------------------------------------------------------------------------
                         TASK 6: Linear and Non Linear SVM.
    ---------------------------------------------------------------------------
    Calculate the SVM with linear and non linear kernel.
    ---------------------------------------------------------------------------                                  
                                                                            '''
    SVML = SVC(kernel='linear')
    SVML.fit(x_train, y_train)
    SVML_y_pred = SVML.predict(x_test)
    accuracySVML = np.mean(y_test == SVML_y_pred)
    print(f'--> Linear SVM Accuracy: {100 * accuracySVML: .2f}%\n')
    
    classifierSVMNL = SVC(kernel='rbf')
    classifierSVMNL.fit(x_train, y_train)
    SVMNL_y_pred = classifierSVMNL.predict(x_test)
    accuracySVMNL = np.mean(y_test == SVMNL_y_pred)
    print(f'--> Non Linear SVM Accuracy: {100 * accuracySVMNL: .2f}%\n')

    '''
    ---------------------------------------------------------------------------
    TASK 7: NEURAL NETWORK
    ---------------------------------------------------------------------------
    Neural Network Implementation with 10 layers: 
    ---------------------------------------------------------------------------
                                                                            '''
    y_train = to_categorical(y_train)
    model = keras.Sequential(layers=[
         Dense(units=15, activation="relu", input_dim=60),
         Dense(units=10, activation="tanh"),
         Dense(units=10, activation="tanh"),
         Dense(units=16, activation="softmax")
     ])
    print(model.summary())
    
    model.compile(optimizer="adam",
                   loss    ="categorical_crossentropy",
                   metrics =["accuracy"])
    history = model.fit(x_train, 
                        y_train, 
                        batch_size=20, 
                        epochs=10)
    NEURAL_y_pred = model.predict(x_test)
    NEURAL_y_pred = np.argmax(NEURAL_y_pred, axis=1)
    print(NEURAL_y_pred)
    print(y_test)
    print(f'--> Accuracy: {np.mean(NEURAL_y_pred == y_test)*100}%')



    x_train_orig, x_test_orig, y_train, y_test = train_test_split(x, 
                                                                  y,
                                                                  test_size=0.3,
                                                                  random_state=42)


    '''
    ---------------------------------------------------------------------------
                       TASK 8: Principal Component Analysis.
    ---------------------------------------------------------------------------
    I have saved the number f column used, the accuracies ofr every column and 
    the information loss. The accuracy is calcolated with the KNN with k = 7.
    Next the list are plotted in two graph:
                                           Numbers of columns and information_loss.
                                           Numbers of columns and accuracies.
    ---------------------------------------------------------------------------
    '''
    accuracies = []
    information_loss = []
    colonne = []
    
    '''
    ---------------------------------------------------------------------------
                                   TASK 9: My PCA
    ---------------------------------------------------------------------------
    Calculate the columns with more than 50000 zeros and remove from the 
    DataFrame
    ---------------------------------------------------------------------------'''
    for components in range(1, x_train_orig.shape[1]):
         print(f'N components: {components}')
    
         pca = PCA(n_components=components)
         x_train = pca.fit_transform(x_train_orig)
         x_test = pca.transform(x_test_orig)
         scaler = MinMaxScaler()
         x_train = scaler.fit_transform(x_train)
         x_test = scaler.transform(x_test)
         print(f'Quantità di informazione: {sum(pca.explained_variance_ratio_) * 100:.2f}%')
         
         KNN = KNeighborsClassifier(n_neighbors=7)
         KNN.fit(x_train, y_train)
         KNN_y_pred = KNN.predict(x_test)
         accuracyKNN = np.mean(y_test == KNN_y_pred)
         
         print(f'KNN Accuracy: {100 * accuracyKNN: .2f}%')
         information_loss.append(sum(pca.explained_variance_ratio_) * 100)
         accuracies.append(accuracyKNN * 100)
         colonne.append(components)

   
    xx = np.arange(2, x_train_orig.shape[1])
    fig = plt.figure()
    fig.patch.set_facecolor('paleturquoise')
    plt.plot(colonne, accuracies, 'indigo')
    plt.xlabel('N of components')
    plt.ylabel('Accuracy')
    plt.title('PCA')
    ax = plt.gca()
    ax.set_facecolor('lightskyblue')
    plt.show()

    fig = plt.figure()
    fig.patch.set_facecolor('paleturquoise')
    plt.plot(colonne, information_loss, 'indigo')
    plt.xlabel('N of components')
    plt.ylabel('Percentage of Information')
    plt.title('PCA')
    ax = plt.gca()
    ax.set_facecolor('lightskyblue')
    plt.show()


    b = ds.columns
    b = b.drop(['Response Id', 'Personality']).values
    
    """
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    New ML Algorithms using myPCA. That consist in drop the columns where 
    0 > 50000. Considering that 0 is the neutral answer, so the more 0 I have
    the less significant is the answer.
    ---------------------------------------------------------------------------
    ---------------------------------------------------------------------------
    """
    values, position = count_unique(b, x)
        
    '''Remove the columns with more than 50000 zeros.'''
    df = ds
    for i in drop_columns:
        df.pop(i)
     

    '''
    ---------------------------------------------------------------------------
              LAST TASK: New Model with 42 columns instead of 60.
    ---------------------------------------------------------------------------
    The accuracy is hight nearby 98-98.50 for the KNN and SVM and 90 for the 
    Neural Network.
    ---------------------------------------------------------------------------
                                                                            '''
    x2 = df.drop(columns=['Personality', 'Response Id']).values
    y2 = df['Personality'].values
    y2 = y2.astype('int')
    
    x_trainsecond, x_testsecond, y_trainsecond, y_testsecond = train_test_split(x2, y2,
                                                         test_size=0.3)
    """
    ---------------------------------------------------------------------------
                                     KNN
    ---------------------------------------------------------------------------
                                                                            """
    KNN2 = KNeighborsClassifier(n_neighbors=7)
    KNN2.fit(x_trainsecond, y_trainsecond)
    KNN_y_pred2 = KNN2.predict(x_testsecond)
    accuracyKNN2 = np.mean(y_testsecond == KNN_y_pred2)
    print(f'--> KNN Accuracy: {100 * accuracyKNN2: .2f}%\n')
    """
    ---------------------------------------------------------------------------
                                 Linear SVM
    ---------------------------------------------------------------------------
                                                                            """
    SVMLsecond = SVC(kernel='linear')
    SVMLsecond.fit(x_trainsecond, y_trainsecond)
    SVML_y_predsecond = SVMLsecond.predict(x_testsecond)
    accuracySVMLsecond = np.mean(y_testsecond == SVML_y_predsecond)
    print(f'--> Linear SVM Accuracy: {100 * accuracySVMLsecond: .2f}%\n')
    """
    ---------------------------------------------------------------------------
                               Non linear SVM
    ---------------------------------------------------------------------------
                                                                            """
    classifierSVMNLsecond = SVC(kernel='rbf')
    classifierSVMNLsecond.fit(x_trainsecond, y_trainsecond)
    SVMNL_y_predsecond = classifierSVMNLsecond.predict(x_testsecond)
    accuracySVMNLsecond = np.mean(y_testsecond == SVMNL_y_predsecond)
    print(f'--> Non Linear SVM Accuracy: {100 * accuracySVMNLsecond: .2f}%\n')
    """
    ---------------------------------------------------------------------------
                               Neural Network
    ---------------------------------------------------------------------------
                                                                            """
    x_trainsecond = np.asarray(x_trainsecond).astype(np.float32)
    y_trainsecond = np.asarray(y_trainsecond).astype(np.float32)
    x_testsecond = np.asarray(x_testsecond).astype(np.float32)
    
    y_trainsecond = to_categorical(y_trainsecond)
    model = keras.Sequential(layers=[
         Dense(units=15, activation="relu", input_dim=42),
         Dense(units=10, activation="tanh"),
         Dense(units=10, activation="tanh"),
         Dense(units=16, activation="softmax")
     ])
    print(model.summary())
    model.compile(optimizer="adam",
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])
    history = model.fit(x_trainsecond, y_trainsecond, batch_size=20, epochs=10)
    NEURAL_y_predsecond = model.predict(x_testsecond)
    NEURAL_y_predsecond = np.argmax(NEURAL_y_predsecond, axis=1)
    print(f'--> Accuracy: {np.mean(NEURAL_y_predsecond == y_testsecond)*100}%\n')

    # -------------------------------------------------------------------------

    modelli = ["KNN k = 3", "KNN k = 5", "KNN k = 7", "Linear SVM", "Non Linear SVM", "Neural Network"]
    results_mean = [98.84, 98.92, 98.94, 95.07, 98.91, 90]
    fig = plt.figure()
    fig.patch.set_facecolor('paleturquoise')

    # ------------------------------------------------------------------------

    plt.bar(modelli, results_mean, 0.5, color='indigo', align='center', )
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on ten attempts after 18 columns elimination")
    ax = plt.gca()
    ax.set_facecolor('lightskyblue')

    plt.show()
