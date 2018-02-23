import tensorflow as tf
from cnn import convolution2d , max_pool , ram , algorithm , convolution2d_manual
from data import type2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define Input
n_classes = 2
x_=tf.placeholder(dtype=tf.float32 , shape=[None , 299,299,3],name ='x_')
y_=tf.placeholder(dtype=tf.float32 , shape=[None],name ='y_')
lr = tf.placeholder(dtype=tf.float32 , shape=None ,name ='lr_')

# Building Layer
layer=x_
pool_indices=[1,4,7,10]
out_chs=[32,32,64,64,64,128,128,128,256,256,256,512,512]
filters=[5,3,5,3,3,3,3,3,3,3,3,3,3]
strides=[2,1,2,1,1,1,1,1,1,1,1,1,1]
#Building Network
assert len(out_chs) == len(filters)
for i in range(13):
    layer = convolution2d('conv_{}'.format(i) , layer , out_chs[i] ,s=1)
    if i in pool_indices:
        'max pool'
        layer=max_pool('maxPool_{}'.format(i), layer)
top_conv=tf.identity(layer , 'top_conv')
logits=ram('ram' ,  top_conv)


# Build Optimizer
pred,pred_cls , cost , train_op,correct_pred ,accuracy = \
    algorithm(y_conv=logits , y_=y_ ,learning_rate=lr , optimizer='sgd' , use_l2_loss=True ,activation='mse')

sess= tf.Session()
init =tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)





if '__main__' == __name__:
    pass;
train_images , train_labels , train_filenames , test_images , test_labels , test_filenames=\
    type2('/Users/seongjungkim/PycharmProjects/fundus/fundus_300_debug' ,onehot=False)
feed_dict = {x_ : train_images[0:1] , y_ : train_labels  , lr:0.01}
_ , loss =sess.run([train_op , cost], feed_dict=feed_dict)



ram_w=sess.run(['ram/w:0']  , feed_dict=feed_dict)
ram_w=np.squeeze(ram_w)
top_conv=sess.run(['top_conv:0']  , feed_dict=feed_dict)
top_conv=np.squeeze(top_conv)
print np.shape(ram_w)
print np.shape(top_conv)
top_conv=top_conv * ram_w
print np.shape(top_conv)
top_conv=np.sum(top_conv , axis=2)
print np.shape(top_conv)
print np.max(top_conv)
print np.min(top_conv)
actmap=Image.fromarray(top_conv*255)
actmap=actmap.resize([299,299] , Image.ANTIALIAS)
plt.imshow(actmap)
plt.show()





# Training






# to visualize Regression Activation map , must be recall weights in RAM layer






