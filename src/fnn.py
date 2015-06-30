__author__ = 'manabchetia'

import pandas as pd
from os.path import join
import numpy as np
from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils

img_dir = '/Users/Apple/Desktop/fer2013'
batch_size = 28709
nb_classes = 7
nb_input_nodes = 48*48
nb_epoch = 1


def load_data(filename):
    df = pd.read_csv(filename, dtype={'emotion':np.int32, 'pixels':str, 'Usage':str})
    df['pixels']=df['pixels'].apply(lambda x: np.fromstring(x,sep=' '))
    return df


def get_train_test(df):
    # Train Data
    df_train = df[df['Usage'] == 'Training']
    X_train = np.vstack(df_train['pixels'])
    y_train = np.asarray(df_train['emotion'])

    # Public Test Data
    df_pbl_test = df[df['Usage'] == 'PublicTest']
    X_pbl_test = np.vstack(df_pbl_test['pixels'])
    #X_pbl_test=np.fromstring(X_pbl_test, dtype=float)
    y_pbl_test = np.asarray(df_pbl_test['emotion'])

    # Private Test Data
    df_pvt_test = df[df['Usage'] == 'PrivateTest']
    X_pvt_test = np.vstack(df_pvt_test['pixels'])
    #X_pvt_test=np.fromstring(X_pvt_test, dtype=float)
    y_pvt_test = np.asarray(df_pvt_test['emotion'])

    return X_train, y_train, X_pbl_test, y_pbl_test, X_pvt_test, y_pvt_test


if __name__ == '__main__':
    filename = 'fer2013.csv'

    # Load Data
    df = load_data(join(img_dir, filename)) #DATA FRAME

    X_train, y_train, X_pbl_test, y_pbl_test, X_pvt_test, y_pvt_test = get_train_test(df)

    print(X_train.shape, 'train samples')
    print(X_pbl_test.shape, 'public test samples')
    print(X_pvt_test.shape, 'private test samples')

    #  PREPROESSING STARTS FOR X_TRAIN ###
    # STEP (1) - SUBTRACT THE MEAN OF EACH IMAGE
    X_train_mean=np.mean(X_train,axis=1)
    print(X_train_mean.shape, 'Mean of Each Training Image')
    #print (X_train_mean)

    X_train_mean=np.tile(X_train_mean,(2304,1))#,axis=0)
    #print (type(X_train))
    #print(X_train_mean.shape, 'Mean of Each Training Image after reshaping')

    X_train_mean=np.transpose(X_train_mean)
    #print(X_train_mean.shape, 'Mean of Each Training Image after reshaping and Transpose')
    X_train=np.subtract(X_train,X_train_mean)


    # STEP(2)- CALCULATE MEAN and standard deviation OVER EACH PIXELS
    X_train_mean11=np.mean(X_train,axis=0)
    print(X_train_mean11.shape, 'Mean of Each Training Image Pixels')
    X_train_mean1=np.tile(X_train_mean11,(28709,1))#,axis=0)
    print(X_train_mean1.shape, 'Mean of Each Training Image Pixels reshaped')
    X_train_std11 = np.std(X_train,axis=0)
    print(X_train_std11.shape, 'Standard Deviation of Each Training Image Pixels')
    X_train_std1=np.tile(X_train_std11,(28709,1))#,axis=0)
    print(X_train_std1.shape, 'Standard Deviation of Each Training Image Pixels reshaped')

    X_train=np.divide(np.subtract(X_train,X_train_mean1),X_train_std1)
    print(X_train.shape,'Training Image after preprocessing')
    print (X_train)


    #Pre-processing of Cross Validation Data
    # STEP (1) - SUBTRACT THE MEAN OF EACH IMAGE
    X_pbl_test_mean=np.mean(X_pbl_test,axis=1)
    print(X_pbl_test_mean.shape, 'Mean of Each Cross Validation Image')
    X_pbl_test_mean=np.tile(X_pbl_test_mean,(2304,1))
    X_pbl_test_mean=np.transpose(X_pbl_test_mean)
    X_pbl_test=np.subtract(X_pbl_test,X_pbl_test_mean)
    print(X_pbl_test.shape,'Cross Validation Image')
    X_pbl_test_mean1 = np.tile(X_train_mean11,(X_pbl_test.shape[0],1))
    X_pbl_test_std1 = np.tile(X_train_std11,(X_pbl_test.shape[0],1))
    X_pbl_test=np.divide(np.subtract(X_pbl_test,X_pbl_test_mean1),X_pbl_test_std1)
    print(X_pbl_test.shape,'Cross Validation Image after preprocessing')


    #Pre-processing of Test Data
    # STEP (1) - SUBTRACT THE MEAN OF EACH IMAGE

    X_pvt_test_mean=np.mean(X_pvt_test,axis=1)
    print(X_pvt_test_mean.shape, 'Mean of Each Cross Validation Image')
    X_pvt_test_mean=np.tile(X_pvt_test_mean,(2304,1))
    X_pvt_test_mean=np.transpose(X_pvt_test_mean)
    X_pvt_test=np.subtract(X_pvt_test,X_pvt_test_mean)
    print(X_pbl_test.shape,'Test Image')
    X_pvt_test_mean1 = np.tile(X_train_mean11,(X_pvt_test.shape[0],1))
    X_pvt_test_std1 = np.tile(X_train_std11,(X_pvt_test.shape[0],1))
    X_pvt_test=np.divide(np.subtract(X_pvt_test,X_pvt_test_mean1),X_pvt_test_std1)
    print(X_pvt_test.shape,'Test Image after preprocessing')
    
    

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_pbl_test = np_utils.to_categorical(y_pbl_test, nb_classes)
    Y_pvt_test = np_utils.to_categorical(y_pvt_test, nb_classes)
    #print((X_train[0]))
    # X_train=X_train.reshape(28709,1,48,48);
    # X_pbl_test=X_pbl_test.reshape(X_pbl_test.shape[0],1,48,48);
    # X_pvt_test=X_pvt_test.reshape(X_pvt_test.shape[0],1,48,48);
    # print(X_train.shape,'After Reshaping')
    # print(X_pbl_test.shape,'After Reshaping')
    # print(X_pvt_test.shape,'After Reshaping')

    ### PRETRAINING AN AUTO ENCODER FOR THE INPUT ONLY
    auto1_batch_size=128;
    auto1_nb_epoch=1;
    auto1=Sequential()
    auto1.add(Dense(2304,1600,init='uniform'))
    auto1.add(Activation('sigmoid'))
    auto1.add(Dense(1600,2304,init='uniform'))
    auto1.add(Activation('tanh'))
    opt=RMSprop()
    auto1.compile(loss='mean_squared_error', optimizer=opt)
    auto1.fit(X_train, X_train, batch_size=auto1_batch_size, 
        nb_epoch=auto1_nb_epoch, show_accuracy=True, verbose=0)
    w=auto1.layers[0].get_weights()
    out1=auto1.predict(x,batch_size=128,verbose=1)


    ## STARTING THE TRAINING OF COVOLUTIONAL NETWORK
    model1_batch_size=128
    model1_nb_epoch=1
    model1 = Sequential()
    model1.add(Dense(2304,1600,weights=w))
    model1.add(Activation('sigmoid'))
    model1.add(Reshape(1,40,40))
    ## INPUT IS 40x40X1
    model1.add(Convolution2D(24, 1,7,7, border_mode='valid')) #34x34x24
    model1.add(Activation('relu'))#34x34x24
    model1.add(MaxPooling2D(poolsize=(2, 2)))#17x17x24
    model1.add(Dropout(0.5))


    model1.add(Convolution2D(32, 24, 5,5, border_mode='valid')) # 13X13x32
    model1.add(Activation('relu'))#13x13x32
    model1.add(MaxPooling2D(poolsize=(2, 2)))#6X6x32
    model1.add(Dropout(0.5))

    model1.add(Convolution2D(48, 32, 3,3, border_mode='valid')) #4x4x48
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(poolsize=(2, 2)))#2x2x48
    model1.add(Dropout(0.5))

    model1.add(Flatten())
    model1.add(Dense(2*2*48, 100))
    model1.add(Activation('sigmoid'))
    model1.add(Dropout(0.5))

    model1.add(Dense(100,nb_classes))
    model1.add(Activation('softmax'))

    ##OPtimizer
    opt=RMSprop()
    ### Compile the loss
    model1.compile(loss='mean_squared_error', optimizer=opt)
    # Training Neural Network
    model1.fit(X_train, Y_train, batch_size=model1_batch_size, nb_epoch=model1_nb_epoch, show_accuracy=True, verbose=2,
              validation_data=(X_pbl_test, Y_pbl_test))
    score = model1.evaluate(X_pvt_test, Y_pvt_test, show_accuracy=True, verbose=0)
    print('Test accuracy:', score[1])
















