#-*- coding:utf-8 -*-
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from utils import plot_images
def load_binary_mnist(data_dir="MNIST_data/" , onehot = True ):
    #load 0,1
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=onehot)

    train_indices_0 = np.where([np.argmax(mnist.train.labels , axis=1) == 0 ])[1]
    train_indices_1 = np.where([np.argmax(mnist.train.labels , axis=1) == 1 ])[1]


    def _get_images(images , labels , ind):
        return images[np.where([np.argmax(labels, axis=1) == ind])[1]]

    def _get_labels(labels, ind):

        mat_=np.zeros([len(np.where([np.argmax(labels, axis=1) == ind])[1]) , 2])
        mat_[:,ind] = 1
        return mat_

    train_0_imgs, val_0_imgs, test_0_imgs, train_1_imgs, val_1_imgs, test_1_imgs = map(
    lambda (images, labels, ind ): _get_images(images, labels, ind), [(mnist.train.images, mnist.train.labels, 0),
                                                              (mnist.validation.images, mnist.validation.labels, 0),
                                                              (mnist.test.images, mnist.test.labels, 0),
                                                              (mnist.train.images, mnist.train.labels, 1),
                                                              (mnist.validation.images, mnist.validation.labels, 1),
                                                              (mnist.test.images, mnist.test.labels, 1)])

    train_0_labs, val_0_labs, test_0_labs, train_1_labs, val_1_labs, test_1_labs = map(
    lambda (labels, ind ): _get_labels(labels, ind), [(mnist.train.labels, 0),
                                                          (mnist.validation.labels, 0),
                                                          (mnist.test.labels, 0),
                                                          (mnist.train.labels, 1),
                                                          (mnist.validation.labels, 1),
                                                          (mnist.test.labels, 1)])

    train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs = map(
    lambda (elements_0, elements_1): np.vstack([elements_0, elements_1]), [(train_0_imgs , train_1_imgs),
                                                                           (train_0_labs , train_1_labs),
                                                                           (val_0_imgs, val_1_imgs),
                                                                           (val_0_labs, val_1_labs),
                                                                           (test_0_imgs, test_1_imgs),
                                                                           (test_0_labs, test_1_labs),])
    return train_imgs, train_labs, val_imgs, val_labs, test_imgs, test_labs
    train_imgs , val_imgs ,test_imgs =map(lambda imgs: imgs.reshape([-1,28,28,1]) , [train_imgs , val_imgs ,test_imgs])


    print np.shape(train_imgs)
    print np.shape(val_imgs)
    print np.shape(train_labs)
    print np.shape(val_labs)
    print np.shape(test_imgs)
    print np.shape(test_labs)
    """
    plot_images(val_imgs[:10])
    plot_images(train_imgs[:10])
    plot_images(train_imgs[-10:])
    plot_images(val_imgs[-10:])
    plot_images(test_imgs[:10])
    plot_images(test_imgs[-10:])
    """


def load_mnist(data_dir="MNIST_data/" , onehot = True):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=onehot)

    return mnist.train.images.reshape([-1,28,28,1]) , mnist.train.labels , \
           mnist.validation.images.reshape([-1,28,28,1]) , mnist.validation.labels, \
           mnist.test.images.reshape([-1,28,28,1]) , mnist.test.labels


def type2(tfrecords_dir, onehot=True, resize=(299, 299) , random_shuffle = True ,limits = [3000 , 1000 , 1000 , 1000] , save_dir_name=None ):
    # normal : 3000
    # glaucoma : 1000
    # retina : 1000
    # cataract : 1000
    train_images, train_labels, train_filenames = [], [], []
    test_images, test_labels, test_filenames = [], [], []

    names = ['normal_0', 'glaucoma', 'cataract', 'retina', 'cataract_glaucoma', 'retina_cataract', 'retina_glaucoma']
    for ind , name in enumerate(names):
        for type in ['train', 'test']:
            imgs, labs, fnames = reconstruct_tfrecord_rawdata(
                tfrecord_path=tfrecords_dir + '/' + name + '_' + type + '.tfrecord', resize=resize)
            print type, ' ', name
            print 'image shape', np.shape(imgs)
            print 'label shape', np.shape(labs)

            if type =='train':
                random_indices = random.sample(range(len(labs)),
                                               len(labs))  # normal , glaucoma , cataract , retina 만 random shuffle 을 한다
                if random_shuffle and ind < 4:
                    print 'random shuffle On : {} limit : {}'.format(name , limits[ind])
                    limit =limits[ind]
                else :
                    limit = None
                train_images.append(imgs[random_indices[:limit]]);
                train_labels.append(labs[random_indices[:limit]]);
                train_filenames.append(fnames[random_indices[:limit]]);
            else :
                test_images.append(imgs);
                test_labels.append(labs);
                test_filenames.append(fnames);
    def _fn1(x, a, b):
        x[a] = np.concatenate([x[a], x[b]], axis=0)  # cata_glau을  cata에 더한다
        return x

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 4), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 4), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 5),
                                                      [train_images, train_labels, train_filenames])  # retina cataract을
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 5), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 5),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 5), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 6), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 6), [test_images, test_labels, test_filenames])

    for i in range(4):
        print '#', np.shape(train_images[i])
    for i in range(4):
        print '#', np.shape(test_images[i])

    train_labels = train_labels[:4]
    train_filenames = train_filenames[:4]

    test_images = test_images[:4]
    test_labels = test_labels[:4]
    test_filenames = test_filenames[:4]

    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = \
        map(lambda x: np.concatenate([x[0], x[1], x[2], x[3]], axis=0), \
            [train_images, train_labels, train_filenames, test_images, test_labels, test_filenames])

    print 'train images ', np.shape(train_images)
    print 'train labels ', np.shape(train_labels)
    print 'train fnamess ', np.shape(train_filenames)
    print 'test images ', np.shape(test_images)
    print 'test labels ', np.shape(test_labels)
    print 'test fnames ', np.shape(test_filenames)
    n_classes = 2
    if onehot:
        train_labels = cls2onehot(train_labels, depth=n_classes)
        test_labels = cls2onehot(test_labels, depth=n_classes)
    if not os.path.isdir('./type2'):
        os.mkdir('./type2')
    if not save_dir_name is None:
        if not os.path.isdir(os.path.join('./type2', save_dir_name)):
            os.mkdir(os.path.join('./type2', save_dir_name))
    count=0
    while True:

        if save_dir_name == None:
            f_path='./type2/{}'.format(count)
        else:
            f_path = os.path.join('./type2',save_dir_name, '{}'.format(count))

        if not os.path.isdir(f_path):
            os.mkdir(f_path)
            break;
        else:
            count += 1



    np.save(os.path.join(f_path , 'train_imgs.npy') , train_images)
    np.save(os.path.join(f_path, 'train_labs.npy'), train_labels)
    np.save(os.path.join(f_path, 'train_fnames.npy'), train_filenames)
    return train_images, train_labels, train_filenames, test_images, test_labels, test_filenames



def reconstruct_tfrecord_rawdata(tfrecord_path, resize=(299, 299)):
    print 'now Reconstruct Image Data please wait a second'
    reconstruct_image = []
    # caution record_iter is generator

    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

    ret_img_list = []
    ret_lab_list = []
    ret_fnames = []
    for i, str_record in enumerate(record_iter):
        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])

        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = example.features.feature['filename'].bytes_list.value[0]
        filename = filename.decode('utf-8')
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        ret_img_list.append(image)
        ret_lab_list.append(label)
        ret_fnames.append(filename)
    ret_imgs = np.asarray(ret_img_list)

    if np.ndim(ret_imgs) == 3:  # for black image or single image ?
        b, h, w = np.shape(ret_imgs)
        h_diff = h - resize[0]
        w_diff = w - resize[1]
        ret_imgs = ret_imgs[h_diff / 2: h_diff / 2 + resize[0], w_diff / 2: w_diff / 2 + resize[1], :]
    elif np.ndim(ret_imgs) == 4:  # Image Up sacle(x) image Down Scale (O)
        b, h, w, ch = np.shape(ret_imgs)
        h_diff = h - resize[0]
        w_diff = w - resize[1]
        ret_imgs = ret_imgs[:, h_diff / 2: h_diff / 2 + resize[0], w_diff / 2: w_diff / 2 + resize[1], :]
    ret_labs = np.asarray(ret_lab_list)
    ret_imgs = np.asarray(ret_imgs)
    ret_fnames = np.asarray(ret_fnames)
    return ret_imgs, ret_labs, ret_fnames



if '__main__' == __name__:
    #load_fundus('/Users/seongjungkim/PycharmProjects/fundus/fundus_300_debug')
    #train_imgs , train_labs , val_imgs , val_labs ,test_imgs , test_labs =load_mnist()
    load_binary_mnist()

