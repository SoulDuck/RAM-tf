import tensorflow as tf
from cnn import convolution2d , max_pool , ram , algorithm , convolution2d_manual , affine
from data import type2 , load_mnist , load_binary_mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import next_batch , validate
import argparse



parser =argparse.ArgumentParser()
parser.add_argument('--data_dir' , type=str  , default='../PycharmProjects/fundus/fundus_300')
parser.add_argument('--data_type' , type=str  , default='fundus')
args=parser.parse_args()
data_dir=args.data_dir

# Load Data
# Fundus
#train_images , train_labels , train_filenames , test_images , test_labels , test_filenames=type2( data_dir ,onehot=False)

# Mnist
train_images , train_labels , validation_images , validation_labels , test_images , test_labels = load_binary_mnist(onehot=False)
h,w ,ch = np.shape(test_images)[1:]
n_classes = np.shape(train_labels)[-1]
# Define Input
x_=tf.placeholder(dtype=tf.float32 , shape=[None , h,w,ch],name ='x_')
y_=tf.placeholder(dtype=tf.float32 , shape=[None , n_classes],name ='y_')
lr = tf.placeholder(dtype=tf.float32 , shape=None ,name ='lr_')

# Building Layer
pool_indices=[1,4,7,10]
out_chs=[32,32,64,64,64,128,128,128,256,256,256,512,512]
filters=[5,3,5,3,3,3,3,3,3,3,3,3,3]
strides=[2,1,2,1,1,1,1,1,1,1,1,1,1]

#for Mnist version
pool_indices=[1,4,7,10]
out_chs=[32,32,64,64,64]
filters=[5,3,5,3,3]
strides=[2,1,2,1,1] #,1,1,1,1,1,1,1,1]


#Building Network
layer=x_
assert len(out_chs) == len(filters)
n_layers=len(out_chs)
for i in range(n_layers):
    layer = convolution2d('conv_{}'.format(i) , layer , out_chs[i] ,s=1)
    if i in pool_indices:
        'max pool'
        layer=max_pool('maxPool_{}'.format(i), layer)

top_conv=tf.identity(layer , 'top_conv')
logits=ram('ram',top_conv)


#logits=ram('ram' ,  top_conv)
# Build Optimizer
pred,pred_cls , cost , train_op,correct_pred ,accuracy = \
    algorithm(y_conv=logits , y_=y_ ,learning_rate=lr , optimizer='sgd' , use_l2_loss=True ,activation='sigmoid' , cost_func='mse')

sess= tf.Session()
init =tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)
max_step = 10000
ckpt_point =100

for step in range(max_step):
    if step % ckpt_point ==0 :
        acc,pred=validate(validation_images , validation_labels , 50 ,sess , pred , cost, x_ ,y_)
        print acc ,pred
    batch_xs, batch_ys =next_batch(train_images ,  train_labels , batch_size=60)
    feed_dict = {x_: batch_xs, y_: batch_ys, lr: 0.01}
    _, loss ,acc = sess.run([train_op, cost ,accuracy], feed_dict=feed_dict)
    print loss ,acc
if '__main__' == __name__:
    pass;






"""
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

"""



# Training






# to visualize Regression Activation map , must be recall weights in RAM layer







