# -*- coding: utf-8 -*-

# -- Sheet 2 --

# Class: CS5100 : Foundations in Artificial Intelligence
# Term: Fall 2021 
# Professor: Christopher Amato
# University: Northeastern University 
# Group: Ankitha Udupa & Darpan Mehra

# Adding libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

# Data acquisition - training and test data 
trainData = pd.read_csv('sign_mnist_train.csv')
testData = pd.read_csv('sign_mnist_test.csv')

print(trainData.shape)
print(testData.shape)
print(trainData.head())

#Converting training and testing data into arrays
train = np.array(trainData, dtype = 'float32')
test = np.array(testData, dtype='float32')

#Labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

#Displaying random image with label
i = random.randint(1,trainData.shape[0])
fig,ax=plt.subplots(figsize=(3,3))
plt.imshow(train[i,1:].reshape((28,28)),cmap='gray') 
label_index = trainData["label"][i]
plt.title(f"{ labels[label_index]}")

#Showing the distribution of data
fig1=plt.figure(figsize=(10,10))
ax1=fig1.add_subplot(211)
trainData['label'].value_counts().plot(kind='bar',ax=ax1)
ax1.set_ylabel('Value Count')
ax1.set_title('Labels')

# Scaling data
X_train = train[:, 1:] / 255
y_train = train[:, 0]
y_train_categorical = to_categorical(y_train, num_classes=25)

X_test = test[:, 1:] / 255
y_test = test[:,0]
y_test_categorical = to_categorical(y_test,num_classes=25)

print(X_train.shape)
print(y_train.shape)

##SVM##
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# #Simple SVC
# svc=SVC().fit(X_train,y_train)

# print("Accuracy on training data",svc.score(X_train,y_train))
# print("Accuracy on test data",svc.score(X_test,y_test))

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train, test_size=0.25,random_state=42)

##DIFFERENT KERNELS AND C
#C=0.005
print("\nC is 0.005")
kernels = ["linear", "rbf", "poly"]
c_1_train=[]
c_1_test=[]
for kernel in kernels:
    svc = SVC(kernel=kernel,C=0.005).fit(X_train, y_train)
    c_1_train.append(svc.score(X_train, y_train) * 100)
    c_1_test.append(svc.score(X_test, y_test) * 100)
    print(f"Accuracy on training data {kernel}", svc.score(X_train, y_train) * 100)
    print(f"Accuracy on test data {kernel}", svc.score(X_test, y_test) * 100)

print(c_1_test[0])
print(c_1_test[1])
print(c_1_test[2])

#C=0.02
print("\nC is 0.02")
kernels = ["linear", "rbf", "poly"]
c_2_train=[]
c_2_test=[]
for kernel in kernels:
    svc = SVC(kernel=kernel,C=0.02).fit(X_train, y_train)
    c_2_train.append(svc.score(X_train, y_train) * 100)
    c_2_test=(svc.score(X_test, y_test) * 100)
    print(f"Accuracy on training data {kernel}", svc.score(X_train, y_train) * 100)
    print(f"Accuracy on test data {kernel}", svc.score(X_test, y_test) * 100)

# #C=0.06
# print("\nC is 0.06")
# kernels = ["linear", "rbf", "poly"]
# c_3_train=[]
# c_3_test=[]
# for kernel in kernels:
#     svc = SVC(kernel=kernel,C=0.06).fit(X_train, y_train)
#     c_3_train.append(svc.score(X_train, y_train) * 100)
#     c_3_test.append(svc.score(X_test, y_test) * 100)
#     print(f"Accuracy on training data {kernel}", svc.score(X_train, y_train) * 100)
#     print(f"Accuracy on test data {kernel}", svc.score(X_test, y_test) * 100)

# C=0.08
print("\nC is 0.08")
c_3_train=[]
c_3_test=[]
for kernel in kernels:
    svc = SVC(kernel=kernel, C=0.08).fit(X_train, y_train)
    c_3_train.append(svc.score(X_train, y_train) * 100)
    c_3_test.append(svc.score(X_test, y_test) * 100)
    print(f"Accuracy on training data {kernel}", svc.score(X_train, y_train) * 100)
    print(f"Accuracy on test data {kernel}", svc.score(X_test, y_test) * 100)

#C=0.2
print("\nC is 0.2")
c_4_train=[]
c_4_test=[]
for kernel in kernels:
    svc = SVC(kernel=kernel, C=0.2).fit(X_train, y_train)
    c_4_train.append(svc.score(X_train, y_train) * 100)
    c_4_test.append(svc.score(X_test, y_test) * 100)
    print(f"Accuracy on training data {kernel}", svc.score(X_train, y_train) * 100)
    print(f"Accuracy on test data {kernel}", svc.score(X_test, y_test) * 100)


# C 0.6
print("\nC is 0.6")
c_5_train=[]
c_5_test=[]
for kernel in kernels:
    svc = SVC(kernel=kernel, C=0.6).fit(X_train, y_train)
    c_5_train.append(svc.score(X_train, y_train) * 100)
    c_5_test.append(svc.score(X_test, y_test) * 100)
    print(f"Accuracy on training data {kernel}", svc.score(X_train, y_train) * 100)
    print(f"Accuracy on test data {kernel}", svc.score(X_test, y_test) * 100)

# C=1
print("\nC is 1")
c_6_train=[]
c_6_test=[]
for kernel in kernels:
    c_6_train.append(svc.score(X_train, y_train) * 100)
    c_6_test.append(svc.score(X_test, y_test) * 100)
    svc = SVC(kernel=kernel, C=1).fit(X_train, y_train)
    print(f"Accuracy on training data {kernel}", svc.score(X_train, y_train) * 100)
    print(f"Accuracy on test data {kernel}", svc.score(X_test, y_test) * 100)

plotAccuracySVM = pd.DataFrame({

    "Linear":[c_1_test[0],c_2_test[0], c_3_test[0], c_4_test[0], c_5_test[0], c_6_test[0]],

    "RBF":[c_1_test[1],c_2_test[1], c_3_test[1], c_4_test[1], c_5_test[1], c_6_test[1]],

    "Poly":[c_1_test[2],c_2_test[2], c_3_test[2], c_4_test[2], c_5_test[2], c_6_test[2]]},

    index=["C=0.005", "C=0.02", "C=0.08", "C=0.2","C=0.6","C=1"])

plotAccuracySVM.plot(kind="bar",figsize=(15, 8))
plt.title("Test Accuracy SVM")
plt.xlabel("Regularization Parameter")
plt.ylabel("Accuracy")





#Convolutional Neural Netwrok
from sklearn.metrics import accuracy_score,confusion_matrix

#Reshaping the data
X_train = X_train.reshape(X_train.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))

print(X_train.shape)
print(y_train.shape)

#ARCHITECTURE 1
#Using 1 Layer
model1 = Sequential()

model1.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Dropout(0.4))

model1.add(Flatten())
model1.add(Dropout(0.4))

model1.add(Dense(128,activation='relu'))
model1.add(Dense(25,activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model1.summary()

history1 = model1.fit(X_train,y_train_categorical, batch_size=128,epochs=10, verbose=1, validation_data=(X_test,y_test_categorical))


#Plotting training and validation loss/accuracy for model 1
loss = history1.history['loss']
ep = range(1,len(loss)+1)

valLoss = history1.history['val_loss']
plt.plot(ep,loss,'y',label='Training Loss')
plt.plot(ep,valLoss,'r',label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history1.history['acc']
valAcc = history1.history['val_acc']

plt.plot(ep,acc,'y',label='Training acc')
plt.plot(ep,valAcc,'r',label='Validation acc')
plt.title("Training and Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Training accuracy of model 1
pred = model1.predict(X_train)
pred = np.argmax(pred,axis=1)
acc_train_model1=accuracy_score(y_train,pred)
print(acc_train_model1)

#Test Accuracy of model 1
pred = model1.predict(X_test)
pred = np.argmax(pred,axis=1)
acc_test_model1=accuracy_score(y_test,pred)
print(acc_test_model1)

#ARCHITECTURE 2
#Increasing Number Of Filters
model2 = Sequential()

model2.add(Conv2D(64,(3,3),input_shape = (28,28,1), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.4))

model2.add(Flatten())
model2.add(Dropout(0.4))

model2.add(Dense(128,activation='relu'))
model2.add(Dense(25,activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model2.summary()

history2 = model2.fit(X_train,y_train_categorical, batch_size=128,epochs=10, verbose=1, validation_data=(X_test,y_test_categorical))

#Plotting training and validation loss/accuracy for model 2
loss = history2.history['loss']
ep = range(1,len(loss)+1)

valLoss = history2.history['val_loss']
plt.plot(ep,loss,'y',label='Training Loss')
plt.plot(ep,valLoss,'r',label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history2.history['acc']
valAcc = history2.history['val_acc']

plt.plot(ep,acc,'y',label='Training acc')
plt.plot(ep,valAcc,'r',label='Validation acc')
plt.title("Training and Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Training accuracy of model 2
pred = model2.predict(X_train)
pred = np.argmax(pred,axis=1)
acc_train_model2=accuracy_score(y_train,pred)
print(acc_train_model2)

#Test Accuracy of model 2
pred = model2.predict(X_test)
pred = np.argmax(pred,axis=1)
acc_test_model2=accuracy_score(y_test,pred)
print(acc_test_model2)

#ARCHITECTURE 3
#Increasing the number of hidden layers
model3 = Sequential()

model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Dropout(0.2))

model3.add(Conv2D(64,(3,3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Dropout(0.2))

model3.add(Flatten())

model3.add(Dense(128,activation='relu'))
model3.add(Dense(25,activation='softmax'))

model3.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
#model3.summary()

history3 = model3.fit(X_train,y_train_categorical, batch_size=128,epochs=10, verbose=1, validation_data=(X_test,y_test_categorical))

#Plotting training and validation loss/accuracy for model 3
loss = history3.history['loss']
ep = range(1,len(loss)+1)

valLoss = history3.history['val_loss']
plt.plot(ep,loss,'y',label='Training Loss')
plt.plot(ep,valLoss,'r',label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history3.history['acc']
valAcc = history3.history['val_acc']

plt.plot(ep,acc,'y',label='Training acc')
plt.plot(ep,valAcc,'r',label='Validation acc')
plt.title("Training and Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Training accuracy of model 3
pred = model3.predict(X_train)
pred = np.argmax(pred,axis=1)
acc_train_model3=accuracy_score(y_train,pred)
print(acc_train_model3)

#Test Accuracy of model 3
pred = model3.predict(X_test)
pred = np.argmax(pred,axis=1)
acc_test_model3=accuracy_score(y_test,pred)
print(acc_test_model3)

#ARCHITECTURE 4
#Increasing Dropout 4
model4 = Sequential()

model4.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation='relu'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Dropout(0.4))

model4.add(Conv2D(64,(3,3), activation='relu'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Dropout(0.4))

model4.add(Flatten())
model4.add(Dropout(0.4))

model4.add(Dense(128,activation='relu'))
model4.add(Dense(25,activation='softmax'))

model4.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model4.summary()

history4 = model4.fit(X_train,y_train_categorical, batch_size=128,epochs=10, verbose=1, validation_data=(X_test,y_test_categorical))

#Plotting training and validation loss/accuracy for model 4
loss = history4.history['loss']
ep = range(1,len(loss)+1)

valLoss = history4.history['val_loss']
plt.plot(ep,loss,'y',label='Training Loss')
plt.plot(ep,valLoss,'r',label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history4.history['acc']
valAcc = history4.history['val_acc']

plt.plot(ep,acc,'y',label='Training acc')
plt.plot(ep,valAcc,'r',label='Validation acc')
plt.title("Training and Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Training accuracy of model 4
pred = model4.predict(X_train)
pred = np.argmax(pred,axis=1)
acc_train_model4=accuracy_score(y_train,pred)
print(acc_train_model4)

#Test Accuracy of model 4
pred = model4.predict(X_test)
pred = np.argmax(pred,axis=1)
acc_test_model4=accuracy_score(y_test,pred)
print(acc_test_model4)

plotAccuracy = pd.DataFrame({

    "train":[acc_train_model1,acc_train_model2, acc_train_model3, acc_train_model4],

    "test":[acc_test_model1,acc_test_model2, acc_test_model3, acc_test_model4]},

    index=["model 1", "model 2", "model 3", "model 4"])

plotAccuracy.plot(kind="bar",figsize=(15, 8))
plt.title("Train and Test Accuracy")
plt.xlabel("Architecture")
plt.ylabel("Accuracy")





