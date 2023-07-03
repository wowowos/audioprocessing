
import os
from keras.models import load_model
import keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np

path = '/root/home/lichaoyang/web/audioprocessing'
def readFile(path):
    # 打开文件（注意路径）
    f = open(path)
    # 逐行进行处理
    lines = f.readlines()
    del lines[0:1589]
    first_ele = True
    for data in lines:
        ## 去掉每行的换行符，"\n"
        data = data.strip('\n')
        ## 按照逗号进行分割。
        nums = data.split(',')
        ## 添加到 matrix 中。
        if first_ele:
            ### 加入到 matrix 中 。
            matrix = np.array(nums)
            first_ele = False
        else:
            matrix = np.c_[matrix,nums]
    matrix = matrix.transpose()
    a = []
    b = [1403,227,1493,290,160,154,1249,97,572,45,89,187,621,334,207,844,608,396,295,205]
    for x in range(0,3):
        result = [float(matrix[2-x][c]) for c in b]
        a.append(result)
    arr=np.array(a)
    f.close()
    return arr

data_test = readFile(os.path.join(path, 'static', 'audio_feature.txt'))  # E:\pythonProject\Audioprocessing\\audio_feature\\audio_feature0{}.txt
for i in range(0, 3):
    for j in range(0, 20):
        if float(data_test[i][j]) != 0.0:
            data_test[i][j] = float(format(data_test[i][j]+0.000000000000001, '.3g'))

health = 0
depression = 0
for i in range(1,4):
    loaded_model = load_model(os.path.join(path, 'static', 'models', 'model_{}.h5'.format(i)))
    loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    if int(np.argmax(loaded_model.predict(np.array(data_test[i-1]).reshape((1,1,20,1))),axis=1)) == 0:
        health += 1
    else :
        depression += 1
    
print(health)
print(depression)