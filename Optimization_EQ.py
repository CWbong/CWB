import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import os 
import cv2
import random
from bayes_opt import BayesianOptimization, UtilityFunction
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout


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

checkpoint_path = "D:/CWB/Ali/202402/2024_Final_Mono_CAE_model2/2024_Final_Mono_CAE_model\\cp.ckpt"
intermediate_Mono =  tf.keras.models.load_model(checkpoint_path)

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
        dense = tf.keras.layers.Dense(units=neurons2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2), name=name)(dense)
        
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


    hist_merged = combined_model.fit(DataTrain_Mono, norm_y2_train, epochs=epochs5,
                       batch_size=batch_size5, shuffle=True,
                       validation_data=(DataValid_Mono, norm_y2_valid),
                       verbose=1)
    
    pre_valid_eq = combined_model.predict(DataValid_Mono)
    pre_valid_eq = pre_valid_eq*(maxeq-mineq)+mineq
    y2_valid_r2 = norm_y2_valid*(maxeq-mineq)+mineq
    
    
    plt.plot(y2_valid_r2, pre_valid_eq,'o', alpha=0.05)
    plt.show()
    diff_eq = y2_valid_r2.reshape(y2_valid.shape[0],1) - pre_valid_eq
    plt.hist(diff_eq)
    plt.show()
    
    
    BO_scores = -1*hist_merged.history['val_loss'][epochs5-1]
    
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
plt.grid(True)
plt.xlabel("conditions", fontsize = 14)
plt.ylabel("Black box function f(x), <-1*rmse>", fontsize = 14)

scores = BO.space.params
scores2 = BO.space.target