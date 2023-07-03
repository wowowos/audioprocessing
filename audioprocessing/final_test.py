# -*- coding: UTF-8 -*-
#包的引入 这些包python自带
import os
from subprocess import call
from django.http import response, HttpResponse, JsonResponse
from django.shortcuts import render
import wave

import numpy as np
from keras.models import load_model
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras import Model, layers
import math

#路径设置
path = '/root/home/lichaoyang/web/audioprocessing'
#SMILExtract_Debug.exe所在的文件路径
pathExcuteFile = os.path.join(path, 'opensmile', 'bin', 'linux_x64_standalone_libstdc6', 'SMILExtract')#不是要求路径具体大小 省略号就是简单省略
#opensmile配置文件所在的路径  一般根据要求会选择不同的配置文件
pathConfig = os.path.join(path, 'opensmile', 'config', 'emobase2010.conf')
pathAudio = os.path.join(path, 'static', 'audios') #E:\pythonProject\Audioprocessing\test该目录下是各个类别文件夹，类别文件夹下才是wav语音文件,比如说，我把wav文件放在了voice的文件夹里，但是voice在new文件夹里  所以应该具体到new文件夹即可，因为下面的代码是对整个文件夹里的所有文件目录里的文件进行操作，具体适用于多种不同类型的语音来进行提取特征
pathOutput = os.path.join(path, 'static')#E:\pythonProject\Audioprocessing这里的路径可以自行设置比如"...\\...\\"python要加一个\转义字符
#利用cmd调用exe文件
def excuteCMD(_pathExcuteFile,_pathConfig,_pathAudio,_pathOutput,_Name):
    cmd = _pathExcuteFile + " -C " + _pathConfig + " -I " + _pathAudio + " -O " + _pathOutput + " -N " + _Name
    call(cmd, shell=True)

def aac2raw(aac_path,raw_path):
    cmd = 'ffmpeg -i ' + aac_path + ' -f s16le -acodec pcm_s16le -ar 44100 -ac 1 ' + raw_path
    call(cmd, shell=True)

def loopExcute(pathwav,patharff):
    for file in sorted(os.listdir(pathwav)):
        if os.path.splitext(file)[1] == '.wav':
            file_path = os.path.join(pathwav,file)
            name = os.path.splitext(file)[0]
            #outputname = 'audio_feature0{}.txt'.format(i)#这里是将所有的特征文件写道一个arff文件里，也可以用一个一直在变的名称来实现一个语音对应一个特征文件
            outputname = 'audio_feature.txt'
            output_path = os.path.join(patharff,outputname)
            excuteCMD(pathExcuteFile, pathConfig, file_path, output_path, name)

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
        result = [float(matrix[x][c]) for c in b]
        a.append(result)
    arr=np.array(a)
    f.close()
    return arr

def test(request):
    context = {}
    context['msg'] = ''
    context['result'] = ''
    i = 0
    ii = 0
    j = 0
    if request.method == 'POST':
        files = request.FILES.getlist('file')
        for file in os.listdir(pathAudio):
            ii += 1
        print(files)
        for f in files:
            j += 1
        #blobfile = request.FILES.get('audiofile')  # 获取上传的音频二进制流的文件对象
            print(f)
            filename1 = '0{}'.format(int(ii/3)+1) + '.aac'
            filename2 = '0{}'.format(int(ii/3)+1) + '.raw'
            wavfilename = '0{}'.format(int(ii/3)+1) + '.wav'
            f_audio = open(os.path.join(pathAudio,filename1),'wb+')
            #f_audio = wave.open(os.path.join(pathAudio,filename),'wb')
            #f_audio.setnchannels(1)
            #f_audio.setnframes(1)
            #f_audio.setsampwidth(2)
            #f_audio.setframerate(44100)
            #with open(f_audio, "wb") as f:
                #for a in files[j]:
                    #f.write(a)
            for chunk in f.chunks():
                #print(chunk)
                #f_audio.writeframes(chunk)
                f_audio.write(chunk)
            f_audio.close()
            
            aac_path = os.path.join(pathAudio,filename1)
            raw_path = os.path.join(pathAudio,filename2)
            aac2raw(aac_path,raw_path)
        
            pcmfile = open(os.path.join(pathAudio,filename2), 'rb')
            pcmdata = pcmfile.read()
            wavfile = wave.open(os.path.join(pathAudio,wavfilename), 'wb')
            wavfile.setframerate(44100)
            wavfile.setsampwidth(2)    #16位采样即为2字节
            wavfile.setnchannels(1)
            wavfile.writeframes(pcmdata)
            wavfile.close()
        
        for file in os.listdir(pathAudio):
            i += 1
        if i == 9:
            #excuteCMD(pathExcuteFile, pathConfig, pathAudio, pathOutput)
            loopExcute(pathAudio, pathOutput)
        
            data_test = readFile(os.path.join(path, 'static', 'audio_feature.txt'))  # E:\pythonProject\Audioprocessing\\audio_feature\\audio_feature0{}.txt
            for i in range(0, 3):
                for j in range(0, 20):
                    if float(data_test[i][j]) != 0.0:
                        data_test[i][j] = float(format(data_test[i][j]+0.000000000000001, '.3g'))
        
            health = 0
            depression = 0
            for i in range(1,4):
                model_path = os.path.join(path, 'static', 'models', 'model_{}.h5'.format(i))
                loaded_model = load_model(model_path)
                loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
                if int(np.argmax(loaded_model.predict(np.array(data_test[i-1]).reshape((1,1,20,1))),axis=1)) == 0:
                    health += 1
                    print("health")
                else :
                    print("depression")
                    depression += 1
        
            context['msg'] = 'ok'
            
            print(health)
            print(depression)
            
            if health >= 2:
                context['result'] = "当前情绪较平静"
            elif depression >= 2:
                context['result'] = "当前情绪较低落"
            
            #os.remove(os.path.join(path, 'static', 'audio_feature.txt'))
            for file in os.listdir(pathAudio):
                os.remove(os.path.join(pathAudio,file))
            os.remove(os.path.join(path, 'static', 'audio_feature.txt'))
    print(context)
    return JsonResponse(context)


def result(request):
    return render(request, "result.html")