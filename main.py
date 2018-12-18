import pickle
import gzip
from PIL import Image
import os
import numpy as np
import scipy.sparse
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def extractDataFromMNIST():
    
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data,validation_data,test_data)

def extractTestUSPSdata():

    USPSMat  = []
    USPSTar  = []
    curPath  = 'USPSdata/Numerals'
    savedImg = []

    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)

    return (USPSMat,USPSTar)

def softmax(x):

    x -= np.max(x)
    return (np.exp(x)).T / (np.sum(np.exp(x),axis= 1)).T

def oneHotEncoding(y):

    val = scipy.sparse.csr_matrix((np.ones(y.shape[0]),(y,np.array(range(y.shape[0])))))
    return np.array(val.todense()).T

def computeLoss(softmax_x, oneHotEncoding_y ,weight, n):

    return ((-1/n) * np.sum(np.dot(oneHotEncoding_y,np.log(softmax_x)))) + (1/2) * np.sum(weight*weight)

def computeGradient(x,softmax_x , oneHotEncoding_y,weight, n):

    difference = oneHotEncoding_y - softmax_x.T

    return (-1/n) * np.dot(x.T,difference ) + weight


    
    
def mulitvariateLogisticRegression(x,y,x_train,y_train):

    x = np.asarray(x)
    y = np.asarray(y)
    weight = np.zeros([x.shape[1],len(np.unique(y))])
    no_Of_Iterations = 300
    learningRate = 0.1 
    losses = []

    for i in range(no_Of_Iterations):

        softmax_x = softmax((np.dot(x,weight)))
        onehotcoding_y = oneHotEncoding(y)
        loss = computeLoss(softmax_x,onehotcoding_y,weight,x.shape[0])
        gradient = computeGradient(x,softmax_x,onehotcoding_y,weight,x.shape[0])
        losses.append(loss)
        weight = weight - (learningRate * gradient)
        print(loss,gradient)

    probs = softmax(np.dot(x_train,weight))
    preds = np.argmax(probs,axis=1)
    accuracy = sum(preds == y_train)/(float(len(y_train)))
    print(accuracy)
    return preds
    #plt.show(losses)

def deepNeuralNetworkImplementation(x_train,y_train,x_test,y_test):

    classifier =  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 2), random_state=1)

    classifier.fit(x_train,y_train)

    y_pred = classifier.predict(x_test)

    print(accuracy_score(y_test, y_pred)*100)

    score = classifier.score(x_test,y_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig('MLPConfusionMatrixUSPS.jpg')

    return y_pred


def cnn_model_fn(x_train,y_train,x_test,y_test,x_valid,y_valid):
    
    batch_size = 64
    epochs = 3
    num_classes = 10

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    x_train = x_train.reshape((-1,28,28,1))
    x_test = x_test.reshape((-1,2,28,1))
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    score = accuracy_score(y_test,y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig('MajorityVotingUSPS.jpg')

    return y_pred


def randomForestClassifier(x_train,y_train,x_test,y_test):

    model = RandomForestClassifier(n_estimators=64, n_jobs=-1) # 0.8827, 29 seconds
    # model = MLPClassifier(max_iter=700) # 0.8557, 190 seconds
    model.fit(x_train, y_train)

    # Predict
    y_pred = model.predict(x_test)
    print(y_pred)
    # Print result
    print(accuracy_score(y_test, y_pred)*100)

    score = model.score(x_test,y_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig('RandomForestConfusionMatricUSPS.jpg')

    return y_pred

def supportVectorMachines(x_train,y_train,x_test,y_test):

    model_linear = SVC(kernel='linear')

    model_linear.fit(x_train,y_train)

    y_pred = model_linear.predict(x_test)

    print(accuracy_score(y_test,y_pred)* 100)

    score = model_linear.score(x_test,y_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig('SVMClassifierUSPSData.jpg')

    return y_pred

    '''model_rbf_with_gamma = SVC(kernel='rbf',gamma=1)

    model_rbf_with_gamma.fit(x_train,y_train)

    y_pred = model_rbf_with_gamma.predict(x_test)

    print(accuracy_score(y_test,y_pred)*100)'''

    '''model_rbf_with_default_values = SVC(kernel='rbf')

    model_rbf_with_default_values.fit(x_train,y_train)

    y_pred = model_rbf_with_default_values.predict(x_test)

    print(accuracy_score(y_test,y_pred)* 100)

    score = model_rbf_with_default_values.score(x_test,y_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig('SVMClassifierwithRBFMNISTData.jpg')'''

def majority_voting(y1,y2,y3,y4,y5):
    array = []
    y = np.zeros((len(y1)))
    for i in range(len(y1)):
        array.append(y1[i])
        array.append(y2[i])
        array.append(y3[i])
        array.append(y4[i])
        y[i] = stats.mode(array).mode[0]
        array = []
    return y



if __name__ == "__main__":

    print("Data Preprocessing ----")
    training_data,validation_data,test_data = extractDataFromMNIST()
    USPSMat,USPSTar = extractTestUSPSdata()
    print("Data Preprocessing Done ----")

    print("Multinomial Logistic Regression")
    # Multiclass Logistic Regression Using Softmax Function

    image , label = training_data[0], training_data[1]
    
    test_image, test_label = test_data[0], test_data[1]

    valid_image, valid_label = validation_data[0], validation_data[1]
    y_pred = mulitvariateLogisticRegression(image,label)
    '''logisticRegr = LogisticRegression()
    logisticRegr.fit(image, label)
    y_pred = logisticRegr.predict(USPSMat)'''

    # DNN Layer on MNIST
    print("DNN on MNIST Data")
    DNN = deepNeuralNetworkImplementation(image,label,USPSMat,USPSTar)

    #CNN Layer on MNIST
    print("CNN layer ON MNIST")

    CNN = cnn_model_fn(image,label,test_image,test_label,valid_image,valid_label)

    #RandomForest

    print("Random Forest Classifier")
    randomForest = randomForestClassifier(image,label,USPSMat, USPSTar)

    print("Support Vector Machines")
    SVM = supportVectorMachines(image,label,USPSMat, USPSTar)
    final_output = majority_voting(y_pred,DNN,randomForest,SVM,SVM)
    score = accuracy_score(test_label,final_output)

    cm = confusion_matrix(test_label, final_output)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig('MajorityVotingUSPS.jpg')


