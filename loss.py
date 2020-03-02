# -*- coding: utf-8 -*-
# @Time    : 2020/2/29 17:28
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : loss.py
# @Software: PyCharm
import tensorflow as tf

def loss_function(model):
    # RPN, class loss
    rpn_cls_score = tf.reshape(model.rpn_cls_score, [-1, 2])
    rpn_label = tf.reshape(model.rpn_labels, [-1])
    rpn_select = tf.where(tf.not_equal(rpn_label, -1))
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
    rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
    rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.cast(rpn_label,tf.int32),tf.cast(rpn_cls_score,tf.float32)))

    # RPN, bbox loss
    rpn_bbox_pred = model.rpn_bbox_pred
    rpn_bbox_targets = model.rpn_bbox_targets
    rpn_bbox_inside_weights = model.rpn_bbox_inside_weights
    rpn_bbox_outside_weights = model.rpn_bbox_outside_weights
    rpn_loss_box = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,rpn_bbox_outside_weights)

    # RCNN, class loss
    cls_score = model.cls_score
    label = tf.reshape(model.labels, [-1])
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.cast(label,tf.int32),tf.cast(tf.reshape(cls_score, [-1, model.num_classes]),tf.float32)))

    # RCNN, bbox loss
    bbox_pred = model.bbox_pred
    bbox_targets = model.bbox_targets
    bbox_inside_weights = model.bbox_inside_weights
    bbox_outside_weights = model.bbox_outside_weights
    loss_box = smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # self._losses['cross_entropy'] = cross_entropy
    # self._losses['loss_box'] = loss_box
    # self._losses['rpn_cross_entropy'] = rpn_cross_entropy
    # self._losses['rpn_loss_box'] = rpn_loss_box

    loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
    #self._losses['total_loss'] = loss

    #self._event_summaries.update(self._losses)
    return loss

    pass

def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.cast(tf.less(abs_in_box_diff, 1.),tf.float32)
    in_loss_box = tf.pow(in_box_diff, 2) * 0.5 * smoothL1_sign + (abs_in_box_diff - 0.5) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_sum(out_loss_box)
    return loss_box
    pass

def log_loss(y_true,y_pred):
    #y_true.shape和y_pred.shape均为(batch_size, n_boxes, n_classes)
    y_pred = tf.math.maximum(y_pred,1e-15)#确保y_pred不为0
    log_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return log_loss
    pass