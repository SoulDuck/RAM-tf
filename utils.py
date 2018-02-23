import numpy as np
import random
import matplotlib.pyplot as plt
import random
import math
def next_batch(imgs, labs, batch_size):
    indices = random.sample(range(np.shape(imgs)[0]), batch_size)
    if not type(imgs).__module__ == np.__name__:  # check images type to numpy
        imgs = np.asarray(imgs)
    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    return batch_xs, batch_ys


def get_acc(true , pred):
    assert np.ndim(true) == np.ndim(pred) , 'true shape : {} pred shape : {} '.format(np.shape(true) , np.shape(pred))
    if np.ndim(true) ==2:
        true_cls =np.argmax(true , axis =1)
        pred_cls = np.argmax(pred, axis=1)

    tmp=[true_cls == pred_cls]
    acc=np.sum(tmp) / float(len(true_cls))
    return acc



def divide_images_labels_from_batch(images, labels ,batch_size):
    debug_flag=False

    batch_img_list=[]
    batch_lab_list = []
    share=len(labels)/batch_size
    #print len(images)
    #print len(labels)
    #print 'share :',share

    for i in range(share+1):
        if i==share:
            imgs = images[i*batch_size:]
            labs = labels[i*batch_size:]
            #print i+1, len(imgs), len(labs)
            batch_img_list.append(imgs)
            batch_lab_list.append(labs)
            if __debug__ ==debug_flag:
                print "######utils.py: divide_images_labels_from_batch debug mode#####"
                print 'total :', len(images), 'batch', i*batch_size ,'-',len(images)
        else:
            imgs=images[i*batch_size:(i+1)*batch_size]
            labs=labels[i * batch_size:(i + 1) * batch_size]
           # print i , len(imgs) , len(labs)
            batch_img_list.append(imgs)
            batch_lab_list.append(labs)
            if __debug__ == debug_flag:
                print "######utils.py: divide_images_labels_from_batch debug mode######"
                print 'total :', len(images) ,'batch' ,i*batch_size ,":",(i+1)*batch_size
    return batch_img_list , batch_lab_list

def validate(val_imgs , val_labs , batch_size ,sess , pred_tensor ,cost_tensor , x_ , y_ ):
    val_imgs_list , val_labs_list= divide_images_labels_from_batch(val_imgs , val_labs , batch_size=batch_size)
    all_preds=[]
    all_costs=[]
    for i , batch_ys in enumerate(val_labs_list):
        batch_xs = val_imgs_list[i]
        preds , costs=sess.run([pred_tensor , cost_tensor] , {x_ :batch_xs  , y_:batch_ys })
        all_preds.extend(preds)
        all_costs.append(costs)
    acc=get_acc( val_labs , all_preds )
    cost = np.mean(all_costs)
    return acc ,cost


def plot_images(imgs , names=None , random_order=False , savepath=None):
    h=math.ceil(math.sqrt(len(imgs)))
    fig=plt.figure()

    for i in range(len(imgs)):
        ax=fig.add_subplot(h,h,i+1)
        if random_order:
            ind=random.randint(0,len(imgs)-1)
        else:
            ind=i
        img=imgs[ind]
        if np.shape(img)[-1] == 1:
            img = np.squeeze(img)

        plt.imshow(img)
        if not names==None:
            ax.set_xlabel(names[ind])
    if not savepath is None:
        plt.savefig(savepath)
    plt.show()

