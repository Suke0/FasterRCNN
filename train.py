# -*- coding: utf-8 -*-
# @Time    : 2020/2/28 8:06
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : train.py
# @Software: PyCharm
import tensorflow as tf
import os,glob
from BatchGenerator import BatchGenerator
from FasterRCNN import FasterRCNN
from loss import loss_function

def train():
    # 定义分类
    LABELS = ["backgroud",
              "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
              "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
              "pottedplant", "sheep", "sofa", "train", "tvmonitor"
              ]

    # LABELS = ["backgroud",
    #         'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter',
    #         'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
    #         'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    #         'wine glass','cup', 'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
    #         'cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
    #         'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    #         ]

    # 获取当前目录
    PROJECT_ROOT = os.path.dirname(__file__)
    weight_dir = os.path.join(PROJECT_ROOT, "weight")

    # 定义样本路径
    train_ann_dir = os.path.join(PROJECT_ROOT, "voc_train_data", "ann", "*.xml")
    train_img_dir = os.path.join(PROJECT_ROOT, "voc_train_data", "img")

    ann_dir = os.path.join(PROJECT_ROOT, "data", "ann", "*.xml")
    img_dir = os.path.join(PROJECT_ROOT, "data", "img")

    val_ann_dir = os.path.join(PROJECT_ROOT, "voc_val_data", "ann", "*.xml")
    val_img_dir = os.path.join(PROJECT_ROOT, "voc_val_data", "img")

    test_img_file = os.path.join(PROJECT_ROOT, "voc_test_data", "img", "*")
    train_test_img_file = os.path.join(PROJECT_ROOT, "voc_train_data", "img", "*")
    batch_size = 1
    # subtract_mean = [123, 117, 104]
    # divide_by_stddev = 128
    # 获取该路径下的xml
    train_ann_fnames = glob.glob(train_ann_dir)
    ann_fnames = glob.glob(ann_dir)
    val_ann_fnames = glob.glob(val_ann_dir)
    test_img_fnames = glob.glob(test_img_file)
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    train_data_generator = BatchGenerator(train_ann_fnames, train_img_dir, LABELS, batch_size,
                                          shuffle=True)
    val_data_generator = BatchGenerator(val_ann_fnames, val_img_dir, LABELS, batch_size,
                                        shuffle=False)

    # 训练并验证
    model = FasterRCNN(True,len(LABELS))
    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = loss_function

    # Prepare the training dataset.


    # Iterate over epochs.
    for epoch in range(10000):
        print('Start of epoch %d' % (epoch,))
        # Iterate over the batches of the dataset.
        for step in range(len(train_ann_fnames)):
            images, imgs_info, ground_truth_boxes = train_data_generator.get()
            with tf.GradientTape() as tape:
                model(tf.cast(images,tf.float32), imgs_info, ground_truth_boxes)
                loss_value = loss_fn(model)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Log every 200 batches.
            if step % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))

            if step % 1000 == 0:
                model.save_weights(weight_dir + '/model_weights.h5')
        # # Display metrics at the end of each epoch.
        # train_acc = train_acc_metric.result()
        # print('Training acc over epoch: %s' % (float(train_acc),))
        # # Reset training metrics at the end of each epoch
        # train_acc_metric.reset_states()
        #
        # # Run a validation loop at the end of each epoch.
        # for x_batch_val, y_batch_val in val_dataset:
        #     val_logits = model(x_batch_val)
        #     # Update val metrics
        #     val_acc_metric(y_batch_val, val_logits)
        # val_acc = val_acc_metric.result()
        # val_acc_metric.reset_states()
        # print('Validation acc: %s' % (float(val_acc),))

    pass

if __name__ == "__main__":
    train()
    pass