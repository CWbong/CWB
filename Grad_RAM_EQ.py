import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import cv2


h = 200
w = 300

CUDA_VISIBLE_DEVICES=""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

tf.keras.backend.clear_session()



fr = [60, 70, 80, 90, 100, 110, 120, 130, 140]
eq = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1];

fr = np.array(fr)
eq = np.array(eq)


numFiles = 50;

len1 = fr.shape[0]
len2 = eq.shape[0]


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

cnt1 = -1
cnt2 = -1
path = "F:/Ali_data/220103/"


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
                
            else:
                cnt2 = cnt2+1
                DataTrain[:,:,cnt2] = img


plt.imshow(img)
plt.show()
           

max11 = np.max(DataTest[:,:,:])
max21 = np.max(DataTrain[:,:,:])

min11 = np.min(DataTest[:,:,:])
min21 = np.min(DataTrain[:,:,:])


maxmax =np.max([max11, max21])

minmin =np.min([min11, min21])



##condition input##

fr = 130
eq = 0.95
num = 50
checkpoint_path_EQ = "F:/CWB/ALI/Ali/202402/2024_Final_EQ_982_0220_model/2024_Final_EQ_982_0220_model//full_model.h5"

model_EQ =  tf.keras.models.load_model(checkpoint_path_EQ)
model_EQ.summary()
combined_model = model_EQ

path = "E:/CWB/ALI/Flame images for CNN/220103/"
data_path = path + str(fr) + "/Mono/" + str(eq)+"/"
file_list = os.listdir(data_path)
grad_ram_nosum = np.zeros(200*300*num).reshape(200,300,num)
mean_grad_ram = np.zeros(200*300).reshape(200,300)

for index in range(0,num):

    file_name = data_path+file_list[index] # 140 0.95 0.65 50 100 60 0.65 50 100
    img_array = np.fromfile(file_name, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img = img[140:340, 75: 375]

    train_img = np.zeros(h*w).reshape(1,h,w,1)
    for ii in range(0,h):
        for jj in range(0,w):
            train_img[0,ii,jj,0] = ((img[ii,jj]-minmin)/(maxmax-minmin))
        
    x =  tf.convert_to_tensor(train_img)
    grad_model = tf.keras.models.Model([combined_model.inputs], [combined_model.get_layer('R_2').output, combined_model.output])

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        last_conv_layer_output, preds = grad_model(x)

    grads = tape.gradient(preds, last_conv_layer_output)
    grads2 = tf.reshape(grads, [10, 15, 16])
    conv_output = tf.reshape(last_conv_layer_output, [10, 15, 16])

    heat_map1 = np.zeros(10*15).reshape(10,15)
    for ii in range(0, 16):
        for jj in range (0,10):
            for kk in range (0,15):
                heat_map1[jj,kk] = heat_map1[jj,kk] + grads2[jj,kk,ii]*conv_output[jj,kk,ii]

    grad_x= np.linspace(1,300,300)
    grad_y= np.linspace(1,200,200)

    grad_ram = cv2.resize(heat_map1,(300,200),interpolation=cv2.INTER_CUBIC)
    grad_ram_nosum[:,:,index] = grad_ram

for ii in range(0,200):
    for jj in range(0,300):
        mean_grad_ram[ii,jj] = np.mean(grad_ram_nosum[ii,jj,:])
        #The average grad-ram image of num images

        
scale = np.max([abs(np.max(mean_grad_ram)), abs(np.min(mean_grad_ram))])

        
grad_ram2 = cv2.rotate(mean_grad_ram, cv2.ROTATE_90_CLOCKWISE)
plt.gca().set_aspect('equal', adjustable='box')
plt.pcolor(grad_ram2, cmap ='jet', vmin = -1*scale, vmax = scale)
plt.colorbar()
#plt.clim(-0.25, 0.13)
plt.savefig('re_EQ_grad_ram' + str(fr)+'_'+str(eq)+'_'+'.tiff', format='tiff', dpi = 300)
#plt.show()


print("finish_condition: "+str(fr)+'_'+str(eq))