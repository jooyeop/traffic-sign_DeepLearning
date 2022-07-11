import pandas as pd
import numpy as np

import os
import pathlib

np.random.seed(42)

#이미지 불러오기
data = (r'E:\codestates\프로젝트\Project4\Meta')
train = (r'E:\codestates\프로젝트\Project4\Train')
test = (r'E:\codestates\프로젝트\Project4\Test')

#변수 설정
data_ = []
labels = []
classes = 43
cur_path = os.getcwd()

#이미지 및 해당 레이블 검색하기
from PIL import Image

for i in range(classes):
    path = os.path.join(train, str(i))
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img) 
            img = Image.open(img_path) 
            img = img.resize((32, 32)) 
            data_.append(np.array(img))
            labels.append(i) 
        except:
            print('error')

#목록을 numpy 배열로 변환
data_ = np.array(data_)
labels = np.array(labels)

#데이터 분할 및 변환
from sklearn.model_selection import train_test_split

#shape 확인
print(data_.shape)
print(labels.shape)

#훈련 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data_, labels, test_size=0.2, random_state=42)

#분할 한 shape 확인
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#레이블을 하나의 핫 인코딩으로 변환
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


#모델 생성 및 컴파일
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

# 모델 구축
model = Sequential() 


# 첫번째 컨볼루션 레이어
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:])) 
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# 두번째 컨볼루션 레이어
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



# 세번째 레이어
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))


# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 모델 확인
model.summary()

#모델훈련
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

#테스트 데이터 정확도 확인
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#분류 모델 시각화 하기
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
start_index = 0
for i in range(43):
    plt.subplot(7, 7, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = model.predict(X_test[start_index:start_index + 1])
    plt.imshow(X_test[start_index].reshape(32, 32, 3), cmap=plt.cm.binary)
    plt.xlabel(np.argmax(prediction[0]))
    start_index += 10
    actual = y_test[start_index - 10:start_index]
    col = 'g'
    if np.argmax(prediction) != np.argmax(actual):
        col = 'r'
    plt.imshow(X_test[start_index].reshape(32, 32, 3), cmap=plt.cm.binary)
    plt.xlabel('{} ({})'.format(np.argmax(actual), np.argmax(prediction)), color=col)
    start_index += 10
plt.show()

#Chance Level 확인
from sklearn.metrics import classification_report


predictions = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42']))




#모델 검증
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
print(accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)))


#CV를 통한 일반화 가능성 확인 K-Fold Cross Validation
from sklearn.model_selection import KFold


kfold = KFold(n_splits=10, shuffle=True, random_state=42)
results = []
for train, test in kfold.split(X_train):
    model.fit(X_train[train], y_train[train])
    results.append(model.evaluate(X_train[test], y_train[test]))


print(results)