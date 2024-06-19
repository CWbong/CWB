import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import os 
import cv2
import random


CUDA_VISIBLE_DEVICES=""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

tf.keras.backend.clear_session()


fr = [60, 70, 80, 90, 100, 110, 120, 130, 140]
eq = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1];
req = [0.607,	0.604,	0.599,	0.598,	0.604,	0.602,	0.6,	0.608,	0.606,
0.646,	0.652,	0.658,	0.645,	0.65,	0.655,	0.659,	0.653,	0.651,
0.704,	0.702,	0.702,	0.697,	0.708,	0.708,	0.707,	0.701,	0.707,
0.741,	0.752,	0.755,	0.748,	0.755,	0.761,	0.75,	0.753,	0.757,
0.799,	0.802,	0.798,	0.801,	0.802,	0.797,	0.808,	0.806,	0.801,
0.858,	0.846,	0.857,	0.852,	0.854,	0.86,	0.854,	0.854,	0.851,
0.897,	0.896,	0.901,	0.904,	0.9,	0.9,	0.912,	0.907,	0.897,
0.955,	0.946,	0.959,	0.949,	0.958,	0.963,	0.954,	0.951,	0.954,
1.007,	1.012,	1.002,	1,	1.002,	1.005,	1.002,	1.001,	1.001]

fr = np.array(fr)
eq = np.array(eq)
req = np.array(req)
req = req.reshape(9,9)

numFiles = 50;

len1 = fr.shape[0]
len2 = eq.shape[0]

h = 200
w = 300

DataTest = []
DataValid = []
DataTrain = []
y1_test = []
y2_test = []
y1_valid = []
y2_valid = []
y1_train = []
y2_train = []
valid_ratio = 0.3
path = "F:/Ali_data/220103/"

print("128line")

for ii in range(0,len1):
    for jj in range(0,len2):
        bTest =0;
        if (ii != 0) and (ii != len1-1) and (jj != 1) and (jj != len2-1):
            if ((((ii%2)) == 0) and (((jj%2)) ==0)) or ((((ii+2)%2) == 1) and (((jj+2)%2)==1)):
                bTest =1
            
        data_path = path + str(fr[ii]) + "/Mono/" + str(eq[jj])+"/"
        file_list = os.listdir(data_path)
        for idx in range(0, numFiles):
            file_name = data_path+file_list[idx]
            img_array = np.fromfile(file_name, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            img = img[140:340, 75: 375]
            
            if bTest ==1:
                valid_score = random.random()
                if valid_ratio >= valid_score:
                    DataValid.append(img)
                    y1_valid.append(fr[ii])
                    y2_valid.append(req[ii,jj])
                else:
                    DataTest.append(img)
                    y1_test.append(fr[ii])
                    y2_test.append(req[ii,jj])
                
            else:
                DataTrain.append(img)
                y1_train.append(fr[ii])
                y2_train.append(req[ii,jj])

DataValid = np.array(DataValid)
y1_valid = np.array(y1_valid)
y2_valid = np.array(y2_valid)

DataTest = np.array(DataTest)
y1_test = np.array(y1_test)
y2_test = np.array(y2_test)

DataTrain = np.array(DataTrain)
y1_train = np.array(y1_train)
y2_train = np.array(y2_train)
           
max11 = np.max(DataTest[:,:,:])
max21 = np.max(DataTrain[:,:,:])
max31 = np.max(DataValid[:,:,:])

min11 = np.min(DataTest[:,:,:])
min21 = np.min(DataTrain[:,:,:])
min31 = np.min(DataValid[:,:,:])


maxmax =np.max([max11, max21, max31])

minmin =np.min([min11, min21, min21])



maxfr = np.max(fr)
minfr = np.min(fr)
maxeq = np.max(req)
mineq = np.min(req)



len2 = DataValid.shape[0]
DataValid_Mono = np.zeros(len2*h*w*1).reshape(len2,h,w,1)
norm_y1_valid = np.zeros(len2)
norm_y2_valid = np.zeros(len2)
for idx in range(0,len2):
    DataValid_Mono[idx,:,:,0] = (DataValid[idx,:,:]-minmin)/(maxmax - minmin)
    norm_y1_valid[idx] = (y1_valid[idx]-minfr)/(maxfr-minfr)
    norm_y2_valid[idx] = (y2_valid[idx]-mineq)/(maxeq-mineq)
del DataValid

len3 = DataTest.shape[0]
DataTest_Mono = np.zeros(len3*h*w*1).reshape(len3,h,w,1)
norm_y1_test = np.zeros(len3)
norm_y2_test = np.zeros(len3)
for idx in range(0,len3):
    DataTest_Mono[idx,:,:,0] = (DataTest[idx,:,:]-minmin)/(maxmax - minmin)
    norm_y1_test[idx] = (y1_test[idx]-minfr)/(maxfr-minfr)
    norm_y2_test[idx] = (y2_test[idx]-mineq)/(maxeq-mineq)

del DataTest
len4 = DataTrain.shape[0]
DataTrain_Mono = np.zeros(len4*h*w).reshape(len4,h,w,1)
norm_y1_train = np.zeros(len4)
norm_y2_train = np.zeros(len4)

for idx in range(0,len4):
    DataTrain_Mono[idx,:,:,0] = (DataTrain[idx,:,:]-minmin)/(maxmax - minmin)
    norm_y1_train[idx] = (y1_train[idx]-minfr)/(maxfr-minfr)
    norm_y2_train[idx] = (y2_train[idx]-mineq)/(maxeq-mineq)

del DataTrain
print("makeImages")

batch_size1= 28.965362248390086
decay_rate1= 0.9275255332199057
decay_steps1= 3.2576935204884108
epochs1= 27.11332796425098
features= 291.36925489966603
initial_learning_rate1= 0.0048729471937919745

train_len = DataTrain_Mono.shape[0]
features = round(features)
batch_size1 = round(batch_size1)
epochs1 = round(epochs1)

def CNNautoencoder_Mono():

    input_R = tf.keras.layers.Input(shape=[h,w,1])
    
    #Encoder
    encoded_R = tf.keras.layers.Conv2D(filters = 8,activation ='relu', kernel_size = [4,4], padding = 'valid', strides = [4,4], name ='R_1')(input_R)
    encoded_R = tf.keras.layers.Conv2D(filters = 16,activation ='relu', kernel_size = [5,5], padding = 'valid',strides = [5,5], name ='R_2')(encoded_R)
    encoded_R = tf.keras.layers.Conv2D(filters = 32,activation ='relu', kernel_size = [5,5], padding = 'valid',strides = [5,5], name ='R_3')(encoded_R)
    encoded_R = tf.keras.layers.Conv2D(filters = features,activation ='relu', kernel_size = [2,3], padding = 'valid', name ='R_4')(encoded_R)
 
   
    decoded_R = tf.keras.layers.Conv2DTranspose(filters = 32,activation ='relu', kernel_size = [2,3], padding = 'valid', name ='R_5')(encoded_R)
    decoded_R = tf.keras.layers.Conv2DTranspose(filters = 16,activation ='relu', kernel_size = [5,5], padding = 'valid',strides = [5,5], name ='R_6')(decoded_R)
    decoded_R = tf.keras.layers.Conv2DTranspose(filters = 8,activation ='relu', kernel_size= [5,5], padding = 'valid',strides = [5,5], name ='R_7')(decoded_R)
    decoded_R = tf.keras.layers.Conv2DTranspose(filters = 1,activation ='relu', kernel_size =[4,4], padding = 'valid',strides = [4,4], name ='R_8')(decoded_R)
    
    return Model(input_R, decoded_R)
    

model_Mono =  CNNautoencoder_Mono()
model_Mono.summary()

decay_steps1 = (train_len/batch_size1)*decay_steps1


lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate1,
    decay_steps=decay_steps1,
    decay_rate=decay_rate1,
    staircase=True,
    name=None)

adam = tf.keras.optimizers.Adam(learning_rate= lr_schedule1)
model_Mono.compile(optimizer= adam, loss='mse', metrics=['mse'])

checkpoint_path_Mono = "2024_Final_Mono_CAE_model\\cp.ckpt"

cp_callback_Mono = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_Mono,
                                                 save_weights_only=False,
                                                 verbose=1)


adam = tf.keras.optimizers.Adam(learning_rate= lr_schedule1)
model_Mono.compile(optimizer= adam, loss='mse', metrics=['mse'])
               
hist_Mono = model_Mono.fit(DataTrain_Mono,DataTrain_Mono, epochs=epochs1,
                    validation_data=(DataValid_Mono,DataValid_Mono),
                   batch_size=batch_size1, shuffle=True,
                   callbacks=[cp_callback_Mono],
                   verbose=1)
