from os.path import join
import numpy as np
import pandas as pd
from keras.preprocessing import image #ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
import pickle
from sklearn.externals import joblib

img_dir = '/Users/Apple/Desktop/fer2013'
nb_classes=7

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
    return X_train, y_train, X_pbl_test, y_pbl_test, X_pvt_test, y_pvt_test


#if __name__ == '__main__':
filename = 'fer2013.csv'
 # Load Data
df = load_data(join(img_dir, filename)) #DATA FRAME
X_train, y_train, X_pbl_test, y_pbl_test, X_pvt_test, y_pvt_test = get_train_test(df)
x=X_train
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_pbl_test = np_utils.to_categorical(y_pbl_test, nb_classes)
Y_pvt_test = np_utils.to_categorical(y_pvt_test, nb_classes)     	
    
#### TRAINING OF WEIGHTS OF FIRST CONVOLUTIONAL LAYER
# INPUT X
model1_batch_size=128
model1_nb_epoch=30
opt=RMSprop()
model1=Sequential()
model1.add(Dense(X_train.shape[1],X_train.shape[1]))
model1.add(Reshape(1,48,48))
model1.add(Convolution2D(12,1,7,7,border_mode='valid'))# CHANGE 3 to 7
model1.add(Convolution2D(1,12,7,7,border_mode='full'))# CHANGE 3 to 7
model1.add(Reshape(2304))
model1.compile(loss='mean_squared_error', optimizer=opt)
model1.fit(X_train,X_train,batch_size=model1_batch_size, nb_epoch=model1_nb_epoch, show_accuracy=True, verbose=0)
out1=model1.predict(X_train,batch_size=128, verbose=1)
print('out1',out1.shape)

# FEEDFORWARD PHASE 1
# INPUT=X
# ## PERFORM POOLING WITH WEIGHTS FROM 1st CONVOLUTIONAL LAYER 
model2=Sequential()
model2.add(Dense(X_train.shape[1],X_train.shape[1],weights=model1.layers[0].get_weights()))
model2.add(Reshape(1,48,48))
model2.add(Convolution2D(12,1,7,7,border_mode='valid',weights=model1.layers[2].get_weights()))
model2.add(MaxPooling2D(poolsize=(2, 2)))
model2.compile(loss='mean_squared_error',optimizer=opt)
out2=model2.predict(X_train,batch_size=128,verbose=1)
print('out2',out2.shape)## 21x21x12
out2r=out2.reshape(X_train.shape[0],out2.shape[1]*out2.shape[2]*out2.shape[3])
print ('out2r',out2r.shape)

### TRAINING OF WEIGHTS OF 2nd CONVOLUtioNAL LAYER
# INPUT out2r
model3_batch_size=128
model3_nb_epoch=30
model3=Sequential()
model3.add(Dense(out2r.shape[1],out2r.shape[1]))
model3.add(Reshape(out2.shape[1],out2.shape[2],out2.shape[3]))
model3.add(Convolution2D(24,12,5,5,border_mode='valid'))
model3.add(Convolution2D(12,24,5,5,border_mode='full'))	
model3.add(Reshape(out2r.shape[1]))
model3.compile(loss='mean_squared_error',optimizer=opt)
model3.fit(out2r,out2r,batch_size=model3_batch_size, nb_epoch=model3_nb_epoch, show_accuracy=True, verbose=0)
out3=model3.predict(out2r,batch_size=128,verbose=1)
print('out3',out3.shape)


## FEEDFORWARD PHASE 2
model4=Sequential()
model4.add(Dense(out2r.shape[1],out2r.shape[1],weights=model3.layers[0].get_weights()))
model4.add(Reshape(out2.shape[1],out2.shape[2],out2.shape[3]))
model4.add(Convolution2D(24,12,5,5,border_mode='valid',weights=model3.layers[2].get_weights()))
model4.add(MaxPooling2D(poolsize=(2,2)))
model4.compile(loss='mean_squared_error',optimizer=opt)
out4=model4.predict(out2r,batch_size=128,verbose=1)
print('out4',out4.shape)## 16x16x24
out4r=out4.reshape(x.shape[0],out4.shape[1]*out4.shape[2]*out4.shape[3])
print ('out4r',out4r.shape) ## 8x8x24

### TRAINING OF WEIGHTS OF 3rd CONVOLUtioNAL LAYER
# INPUT out4r

model5_batch_size=128
model5_nb_epoch=30
model5=Sequential()
model5.add(Dense(out4r.shape[1],out4r.shape[1]))
model5.add(Reshape(out4.shape[1],out4.shape[2],out4.shape[3]))
model5.add(Convolution2D(32,24,3,3, border_mode='valid'))
model5.add(Convolution2D(24,32,3,3, border_mode='full'))
model5.add(Reshape(out4r.shape[1]))
model5.compile(loss='mean_squared_error',optimizer=opt)
model5.fit(out4r,out4r,batch_size=model5_batch_size, nb_epoch=model5_nb_epoch, show_accuracy=True, verbose=0)
out5=model5.predict(out4r,batch_size=128,verbose=1)
print('out5',out5.shape)


## FEEDFORWARD PHASE 3
model6=Sequential()
model6.add(Dense(out4r.shape[1],out4r.shape[1],weights=model5.layers[0].get_weights()))
model6.add(Reshape(out4.shape[1],out4.shape[2],out4.shape[3]))
model6.add(Convolution2D(32,24,3,3,border_mode='valid',weights=model5.layers[2].get_weights()))
model6.add(MaxPooling2D(poolsize=(2,2)))
model6.compile(loss='mean_squared_error',optimizer=opt)
out6=model6.predict(out4r,batch_size=128,verbose=1)
print('out6',out6.shape)## 6x6x32
out6r=out6.reshape(x.shape[0],out6.shape[1]*out6.shape[2]*out6.shape[3])
print ('out6r',out6r.shape) ## 3x3x24


## FINAL TRAINING CONVOLUTIONAL NETWORK
modelF_batch_size=128
modelF_nb_epoch=30
modelF=Sequential()
modelF.add(Dense(x.shape[1],x.shape[1])) ## FIRST DENSE LAYER FROM model1
modelF.add(Reshape(1,48,48)) ## Reshaping from first layer

modelF.add(Convolution2D(12,1,7,7,border_mode='valid',weights=model1.layers[2].get_weights())) #41X41X12
modelF.add(MaxPooling2D(poolsize=(2, 2))) # 21x21x12
modelF.add(Dropout(0.3))

modelF.add(Convolution2D(24,12,5,5,border_mode='valid',weights=model3.layers[2].get_weights())) # 17X17X24
modelF.add(MaxPooling2D(poolsize=(2, 2))) # 8X8X24
modelF.add(Dropout(0.3))

modelF.add(Convolution2D(32,24,3,3,border_mode='valid',weights=model5.layers[2].get_weights())) # 6x6x32
modelF.add(MaxPooling2D(poolsize=(2,2))) # 3x3x32
modelF.add(Dropout(0.3))

modelF.add(Flatten())
model1.add(Dense(288, 100))
model1.add(Activation('sigmoid'))
model1.add(Dropout(0.3))
model1.add(Dense(100,nb_classes))
model1.add(Activation('softmax'))
modelF.compile(loss='mean_squared_error',optimizer=opt)
modelF.fit(X_train, Y_train, batch_size=modelF_batch_size, nb_epoch=modelF_nb_epoch, show_accuracy=True, verbose=2,validation_data=(X_pbl_test, Y_pbl_test))
outF=modelF.predict(X_train,batch_size=128,verbose=1)
print('outF',outF.shape)



model1.save_weights('../fer2013/model/model1weights.h5py',overwrite=True)
model2.save_weights('../fer2013/model/model2weights.h5py',overwrite=True)
model3.save_weights('../fer2013/model/model3weights.h5py',overwrite=True)
model4.save_weights('../fer2013/model/model4weights.h5py',overwrite=True)
model5.save_weights('../fer2013/model/model5weights.h5py',overwrite=True)
model6.save_weights('../fer2013/model/model6weights.h5py',overwrite=True)
modelF.save_weights('../fer2013/model/modelFweights.h5py',overwrite=True)


