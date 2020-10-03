import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time
from custom_tqdm import TqdmNotebookCallback
from tqdm.keras import TqdmCallback
import albumentations as A
import random
import io
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import cv2
from pathlib import Path
import os

class ClassifierModel(keras.Model):
    """ClassifierModel
    Predicts logits.
    Takes raw input in uint8 dtype.
    
    Output
    ------
    logits : tf.Tensor
    """
    def __init__(self, inputs, model_function):
        """
        Because of numerical stability, softmax layer should be
        taken out, and use it only when not training.
        Args
            inputs : keras.Input
            model_function : function that takes keras.Input and returns
            output tensor of logits
        """
        super().__init__()
        outputs = model_function(inputs)
        self.logits = keras.Model(inputs=inputs, outputs=outputs)
        self.logits.summary()
        
    def call(self, inputs, training=None):
        casted = tf.cast(inputs, tf.float32) / 255.0
        return self.logits(inputs, training=training)

class AugGenerator():
    """An iterable generator that makes augmented ImageNet image data

    NOTE: 
        Every img is reshaped to img_size

    return
    ------
    X : np.array, dtype= np.uint8
        shape : (HEIGHT, WIDTH, 3)
    Y : np.array, dtype= np.float32
    """
    def __init__(self, img_dir, img_names, label_dict, img_size):
        """ 
        arguments
        ---------
        img_dir : str
            path to the image directory
        img_names : list
            list of image names. img_dir/img_name should be the full path
            image name should be 
        label_dict : dict
            dictionary mapping from ID -> category number
        img_size : tuple
            Desired output image size
            IMPORTANT : (HEIGHT, WIDTH)
        """
        self.img_dir = img_dir
        self.img_names = img_names
        self.label_dict = label_dict
        
        # Find label numbers prior to save time
        self.img_labels = \
            [self.label_dict[n.split('_')[0]] for n in self.img_names]

        self.n = len(img_names)
        self.output_size = img_size
        self.aug = A.Compose([
            A.OneOf([
                A.RandomGamma((40,200),p=1),
                A.RandomBrightness(limit=0.5, p=1),
                A.RandomContrast(limit=0.5,p=1),
                A.RGBShift(40,40,40,p=1),
                A.Downscale(scale_min=0.25,scale_max=0.5,p=1),
                A.ChannelShuffle(p=1),
            ], p=0.8),
            A.InvertImg(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            A.Resize(img_size[0], img_size[1]),
            A.Cutout(8,img_size[0]//12,img_size[1]//12)
        ],
        )

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        idx = random.randrange(0,self.n)

        image_name = self.img_names[idx]
        label = self.img_labels[idx]
        full_path = os.path.join(self.img_dir,image_name)
        image = cv2.cvtColor(cv2.imread(full_path,cv2.IMREAD_COLOR),
                                        cv2.COLOR_BGR2RGB)
        
        distorted = self.aug(
            image=image,
        )

        return distorted['image'], label

class ValGenerator(AugGenerator):
    """Same as AugGenerator, but without augmentation.
    Only resizes the image
    """
    def __init__(self, img_dir, img_names, label_dict, img_size):
        """ 
        arguments
        ---------
        img_dir : str
            path to the image directory
        img_names : list
            list of image names. img_dir/img_name should be the full path
            image name should be 
        label_dict : dict
            dictionary mapping from ID -> category number
        img_size : tuple
            Desired output image size
            IMPORTANT : (WIDTH, HEIGHT)
        """
        super().__init__(img_dir, img_names, label_dict, img_size)
        self.aug = A.Resize(img_size[0], img_size[1])

def create_train_dataset(
        img_dir, 
        img_names, 
        label_dict, 
        img_size, 
        batch_size, 
        buffer_size=1000,
        val_data=False):
    autotune = tf.data.experimental.AUTOTUNE
    if val_data:
        generator = ValGenerator(
            img_dir,
            img_names,
            label_dict,
            img_size,
        )
    else:
        generator = AugGenerator(
            img_dir,
            img_names,
            label_dict,
            img_size,
        )
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.uint8, tf.int64),
        output_shapes=(
            tf.TensorShape([img_size[0],img_size[1],3]), 
            tf.TensorShape([])
        ),
    )
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset

def imagenet_val_dataset(
        img_dir, 
        true_labels, 
        img_size,
        batch_size,
        buffer_size=1000,
    ):
    autotune = tf.data.experimental.AUTOTUNE
    img_names = sorted(os.listdir(img_dir))
    img_full = [os.path.join(img_dir,n) for n in img_names]

    dataset = tf.data.Dataset.from_tensor_slices((img_full,true_labels))

    dataset = dataset.map(partial(parse_image,img_size=img_size))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset

def parse_image(filename, label, img_size=None):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image,tf.uint8)
    return image, label

def get_model(model_f, img_size):
    """
    To get model only and load weights.
    """
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    inputs = keras.Input((img_size[0],img_size[1],3))
    test_model = ClassifierModel(inputs, model_f)
    test_model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(),
        ]
    )
    return test_model

class ValFigCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, logdir, label_names):
        super().__init__()
        self.val_ds = val_ds
        self.filewriter = tf.summary.create_file_writer(logdir+'/val_image')
        self.label_names = label_names

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def val_result_fig(self):
        sample = self.val_ds.take(1).as_numpy_iterator()
        sample = next(sample)
        sample_x = sample[0]
        sample_y = sample[1]
        predict = self.model(sample_x, training=False).numpy()
        fig = plt.figure(figsize=(15,15))
        for i in range(5):
            ax = fig.add_subplot(5,1,i+1)
            img = sample_x[i]
            ax.imshow(img)
            ax.title.set_text(self.label_names[np.argmax(predict[i])])
        return fig

    def on_epoch_end(self, epoch, logs=None):
        image = self.plot_to_image(self.val_result_fig())
        with self.filewriter.as_default():
            tf.summary.image('val prediction', image, step=epoch)

def run_training(
        model_f, 
        lr_f, 
        name, 
        epochs, 
        batch_size, 
        train_dir,
        label_dict,
        val_dir,
        val_labels,
        id_to_name,
        img_size,
        mixed_float = True,
        notebook = True,
        load_model_path = None,
        profile = False,
    ):
    """
    val_data : (X_val, Y_val) tuple
    """
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    
    st = time.time()

    inputs = keras.Input((img_size[0],img_size[1],3))
    mymodel = ClassifierModel(inputs, model_f)
    if load_model_path:
        mymodel.load_weights(load_model_path)
        print('loaded from : ' + load_model_path)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mymodel.compile(
        optimizer='adam',
        loss=loss,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        ]
    )

    logdir = 'logs/fit/' + name
    if profile:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            profile_batch='3,5',
            update_freq='epoch'
        )
    else :
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            profile_batch=0,
            update_freq='epoch'
        )
    lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

    savedir = 'savedmodels/' + name + '/{epoch}'
    save_callback = keras.callbacks.ModelCheckpoint(
        savedir,
        save_weights_only=True,
        verbose=1
    )

    if notebook:
        tqdm_callback = TqdmNotebookCallback(metrics=['loss','accuracy'],
                                            leave_inner=False)
    else:
        tqdm_callback = TqdmCallback()

    train_names = os.listdir(train_dir)

    train_ds = create_train_dataset(
        train_dir,
        train_names,
        label_dict,
        img_size,
        batch_size,
    )
    val_ds = imagenet_val_dataset(
        val_dir,
        val_labels,
        img_size,
        batch_size,
    )

    image_callback = ValFigCallback(val_ds, logdir, id_to_name)

    mymodel.fit(
        x=train_ds,
        epochs=epochs,
        steps_per_epoch=len(train_names)//batch_size,
        # steps_per_epoch=10,
        callbacks=[
            tensorboard_callback,
            lr_callback,
            save_callback,
            tqdm_callback,
            image_callback,
        ],
        verbose=0,
        validation_data=val_ds,
        validation_steps=100,
    )


    print('Took {} seconds'.format(time.time()-st))

if __name__ == '__main__':
    import os
    import imageio as io
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import draw
    import cv2
    from pathlib import Path

    data_dir = Path('data')
    data_groups = next(os.walk(data_dir))[1]
    img = []
    data = []
    img_name_dict = {}
    img_idx = 0
    for dg in data_groups[:]:
        img_dir = data_dir/dg/'done'
        img_names = os.listdir(img_dir)
        for name in img_names:
            img_path = str(img_dir/name)
            img.append(io.imread(img_path))
            img_name_dict[img_path] = img_idx
            img_idx += 1

        json_dir = data_dir/dg/'save'
        json_names = os.listdir(json_dir)
        dg_data = []
        for name in json_names[:]:
            with open(str(json_dir/name),'r') as j:
                dg_data.extend(json.load(j))
        for dg_datum in dg_data :
            long_img_name = str(img_dir/dg_datum['image'])
            dg_datum['image'] = img_name_dict[long_img_name]
        data.extend(dg_data)

    # fig = plt.figure()
    # d_idx = random.randrange(0,len(data)-5)
    # for i, d in enumerate(data[d_idx:d_idx+5]):
    #     image = img[d['image']].copy()
    #     image = cv2.resize(image, (1200,900), interpolation=cv2.INTER_LINEAR)
    #     mask = d['mask']
    #     m_idx = random.randrange(0,len(mask[0]))
    #     pos = (mask[0][m_idx], mask[1][m_idx])
    #     boxmin = d['box'][0]
    #     boxmax = d['box'][1]
    #     rr, cc = draw.disk((pos[1],pos[0]),5)
    #     image[rr, cc] = [0,255,0]
    #     rr, cc = draw.rectangle_perimeter((boxmin[1],boxmin[0]),(boxmax[1],boxmax[0]))
    #     image[rr,cc] = [255,0,0]
    #     image[mask[1],mask[0]] = [100,100,100]
    #     ax = fig.add_subplot(5,1,i+1)
    #     ax.imshow(image)
    # plt.show()

    # gen = AugGenerator(img, data, (400,400))
    # s = next(gen)

    ds = create_train_dataset(img, data, (200,200),1, False)
    sample = ds.take(5).as_numpy_iterator()
    fig = plt.figure()
    for i, s in enumerate(sample):
        ax = fig.add_subplot(5,2,2*i+1)
        img = s[0][0].swapaxes(0,1)
        ax.imshow(img)
        ax = fig.add_subplot(5,2,2*i+2)
        mask = s[1][0].swapaxes(0,1)
        ax.imshow(mask)
    plt.show()