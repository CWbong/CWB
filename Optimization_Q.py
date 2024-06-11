import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import os 
import cv2
import random
from bayes_opt import BayesianOptimization, UtilityFunction


CUDA_VISIBLE_DEVICES=""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

tf.keras.backend.clear_session()


"""
cv2.imshow('color_img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

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

indices = np.zeros(len2,)

testJJ  = np.ones(len1,)
for ii in range(0,len2):
    indices[ii] = ii

random.shuffle(indices)

for ii in range(0,testJJ.shape[0]):
    testJJ[ii] = indices[(ii%len2)]

cnt1 = 0
cnt2 = 0


for ii in range(0,len1):
    for jj in range(0,len2):
        bTest =0;
        if (ii != 0) and (ii != len1-1) and (jj != 1) and (jj != len2-1):
            if ((((ii%2)) == 0) and (((jj%2)) ==0)) or ((((ii+2)%2) == 1) and (((jj+2)%2)==1)):
                bTest =1
        
        for idx in range(0, numFiles):
            if bTest ==1:
                cnt1 = cnt1+1
            else:
                cnt2 = cnt2+1

len3 = cnt1
len4 = cnt2
h = 200
w = 300

DataTest = np.zeros(len3*h*w).reshape(h,w,len3)
DataTrain = np.zeros(len4*h*w).reshape(h,w,len4)
y1_test = np.zeros(len3)
y2_test = np.zeros(len3)
y1_train = np.zeros(len4)
y2_train = np.zeros(len4)

cnt1 = -1
cnt2 = -1
path = "D:/CWB/Flame images for CNN/220103/"

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
                cnt1 = cnt1+1
                DataTest[:,:,cnt1] = img
                y1_test[cnt1] = fr[ii]
                y2_test[cnt1] = req[ii,jj]
                
            else:
                cnt2 = cnt2+1
                DataTrain[:,:,cnt2] = img
                y1_train[cnt2] = fr[ii]
                y2_train[cnt2] = req[ii,jj]
                
plt.imshow(img)
plt.show()
           
max11 = np.max(DataTest[:,:,:])
max21 = np.max(DataTrain[:,:,:])

min11 = np.min(DataTest[:,:,:])
min21 = np.min(DataTrain[:,:,:])


maxmax =np.max([max11, max21])
minmin =np.min([min11, min21])



maxfr = np.max(fr)
minfr = np.min(fr)
maxeq = np.max(req)
mineq = np.min(req)


print("205line")

normTest = np.zeros(len3*h*w*1).reshape(h,w,len3)
norm_y1_test = np.zeros(len3)
norm_y2_test = np.zeros(len3)
for idx in range(0,len3):
    normTest[:,:,idx] = (DataTest[:,:,idx]-minmin)/(maxmax - minmin)
    norm_y1_test[idx] = (y1_test[idx]-minfr)/(maxfr-minfr)
    norm_y2_test[idx] = (y2_test[idx]-mineq)/(maxeq-mineq)
    
normTrain = np.zeros(len4*h*w).reshape(h,w,len4)
norm_y1_train = np.zeros(len4)
norm_y2_train = np.zeros(len4)
print("219line")

for idx in range(0,len4):
    normTrain[:,:,idx] = (DataTrain[:,:,idx]-minmin)/(maxmax - minmin)
    norm_y1_train[idx] = (y1_train[idx]-minfr)/(maxfr-minfr)
    norm_y2_train[idx] = (y2_train[idx]-mineq)/(maxeq-mineq)
    
print("228line")

DataTrain_Mono = np.zeros(len4*h*w*1).reshape(len4,h,w,1)
DataTest_Mono = np.zeros(len3*h*w*1).reshape(len3,h,w,1)

for index in range(0,len4):
    for ii in range(0,h):
        for jj in range(0,w):
            DataTrain_Mono[index,ii,jj,0] = normTrain[ii,jj,index]
            
for index in range(0,len3):
    for ii in range(0,h):
        for jj in range(0,w):
            DataTest_Mono[index,ii,jj,0] = normTest[ii,jj,index]
            
print("236line")

index = np.linspace(0,DataTest_Mono.shape[0]-1,DataTest_Mono.shape[0])
random.shuffle(index)

del DataTest, DataTrain

len_valid = int(index.shape[0]*0.3)
len_test = int(index.shape[0] - len_valid)
print("261line")

Data_norm_valid = np.zeros(h*w*len_valid).reshape(len_valid,h,w,1)
Data_norm_test = np.zeros(h*w*len_test).reshape(len_test,h,w,1)

for ii in range(0, len_valid):
    Data_norm_valid[ii,:,:,0] = DataTest_Mono[int(index[ii]),:,:,0]


cnt44 = 0
for ii in range(len_valid, len3):    
    Data_norm_test[cnt44,:,:,0] = DataTest_Mono[int(index[ii]),:,:,0]

    cnt44 = cnt44+1
print("277line")

plt.imshow(Data_norm_test[100,:,:,0],cmap ='gray')
plt.show()

y1_valid = np.zeros(len_valid)
y2_valid = np.zeros(len_valid)

split_y1_test = np.zeros(len_test)
split_y2_test = np.zeros(len_test)
for ii in range(0, len_valid):
    y1_valid[ii] = norm_y1_test[int(index[ii])]
    y2_valid[ii] = norm_y2_test[int(index[ii])]
    
Valid_raw = np.zeros(len_valid*h*w).reshape(len_valid,h,w)

for ii in range(0, len_valid):
    Valid_raw[ii,:,:] = (Data_norm_valid[ii,:,:,0]*(maxmax - minmin))+minmin
    
print("290line")
checkpoint_path = "D:/CWB/Ali/202402/2024_Final_Mono_CAE_model2/2024_Final_Mono_CAE_model\\cp.ckpt"
model_Mono =  model = tf.keras.models.load_model(checkpoint_path)
model_Mono.summary()

intermediate_Mono = model_Mono

Valid_recon =intermediate_Mono.predict(Data_norm_valid)

Valid_recon_raw = np.zeros(len_valid*h*w).reshape(len_valid,h,w)

for ii in range(0, len_valid):
    Valid_recon_raw[ii,:,:] = (Valid_recon[ii,:,:,0]*(maxmax - minmin))+minmin

for layer in intermediate_Mono.layers:
    layer.trainable = False

train_len = DataTrain_Mono.shape[0]

def BlackBoxFn(neurons2,L2, initial_learning_rate5, decay_steps5, decay_rate5, batch_size5, epochs5,num_dense_layers):
    neurons2 = round(neurons2)
    batch_size5 = round(batch_size5)
    epochs5 = round(epochs5)
    num_dense_layers = round(num_dense_layers)
    
    decay_steps5 = (train_len/batch_size5)*decay_steps5

    encoded_Mono = intermediate_Mono.get_layer('R_4').output
    flatten = tf.keras.layers.Flatten()(encoded_Mono)
    
    dense = tf.keras.layers. Dense(units=neurons2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2), name='dense_layer1')(flatten)
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        #name2 = 'layer_batch_{0}'.format(i+1)
        #batch = tf.keras.layers.BatchNormalization(name=name2)(dense) 
        dense = tf.keras.layers.Dense(units=neurons2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2), name=name)(dense)
        
    #batch = tf.keras.layers.BatchNormalization(name='batch_last')(dense)
    output_layer = tf.keras.layers.Dense(units=1, activation='relu', name='output_layer')(dense)
    combined_model = Model(inputs =intermediate_Mono.input,
                           outputs = output_layer)
    combined_model.summary()
    plot_model(combined_model, to_file='ae.png', show_shapes=True)
    
    lr_schedule5 = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate5,
        decay_steps=decay_steps5,
        decay_rate=decay_rate5,
        staircase=True,
        name=None)

    adam5 = tf.keras.optimizers.Adam(learning_rate= lr_schedule5)


    combined_model.compile(optimizer= adam5, loss='mse', metrics=['mse'])


    hist_merged = combined_model.fit(DataTrain_Mono, norm_y1_train, epochs=epochs5,
                       batch_size=batch_size5, shuffle=True,
                       validation_data=(Data_norm_valid, y1_valid),
                       verbose=1)
    
    pre_valid_fr = combined_model.predict(Data_norm_valid)
    pre_valid_fr = pre_valid_fr*(maxfr-minfr)+minfr
    y1_valid_r2 = y1_valid*(maxfr-minfr)+minfr
    
    
    plt.plot(y1_valid_r2, pre_valid_fr,'o', alpha=0.05)
    plt.show()
    diff_fr = y1_valid_r2.reshape(y1_valid.shape[0],1) - pre_valid_fr
    plt.hist(diff_fr)
    plt.show()
    
    
    BO_scores = -1*hist_merged.history['val_loss'][epochs5-1]
    print(BO_scores)   

    return BO_scores

pbounds = {'neurons2': (50, 600),
           'L2' : (0, 0.00005),
           'initial_learning_rate5': (0.00001, 0.01),
           'batch_size5': (4, 32),
           'decay_steps5': (0.1, 10),
           'decay_rate5': (0.5, 0.96),
           'epochs5': (5, 30),
           'num_dense_layers': (0, 3)}

utility = UtilityFunction(kind = "ei", xi = 0.01)
BO = BayesianOptimization(f = BlackBoxFn,
                                 pbounds = pbounds, verbose = 1,
                                 random_state = None)

search_num = 100
for i in range(search_num):
    next_point = BO.suggest(utility)
    target = BlackBoxFn(**next_point)
    try:
        BO.register(params = next_point, target = target)
    except:
        pass # Error evasion

num_bo = np.zeros(search_num).reshape(search_num,1)
for ii in range(0, search_num):
    num_bo[ii,0] = 1+ii


print("Best result: {}; f(x) = {:.3f}.".format(BO.max["params"], BO.max["target"]))
plt.figure(figsize = (15, 8))
plt.plot(num_bo, BO.space.target, "o")
# plt.plot(range(1, 1 + len(BO.space.target)), BO.space.target, "-o")
plt.grid(True)
plt.xlabel("conditions", fontsize = 14)
plt.ylabel("Black box function f(x), <-1*rmse>", fontsize = 14)

scores = BO.space.params
#name = BO.space.pbounds
scores2 = BO.space.target

