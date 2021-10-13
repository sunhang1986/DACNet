"""
	Based on the PointNet++ codebase 
    https://github.com/charlesq34/pointnet2
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import tf_util_srn
import tensorflow.contrib.slim as slim

def CoordAtt(x, reduction = 32):
    def coord_act(x):
        tmpx = tf.nn.relu6(x+3) / 6
        x = x * tmpx
        return x

    x_shape = x.get_shape().as_list()
    [b, h, w, c] = x_shape
    print("b,h,w,c", b, h, w, c)
    x_h = slim.avg_pool2d(x, kernel_size = [1, w], stride = 1)
    print("x_h", x_h.shape)
    x_w = slim.avg_pool2d(x, kernel_size = [h, 1], stride = 1)
    print("x_w", x_w.shape)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    print("x_w", x_w.shape)

    y = tf.concat([x_h, x_w], axis=1)
    print("y", y.shape)
    mip = max(8, c // reduction)
    print("mip", mip)
    y = slim.conv2d(y, mip, (1, 1), stride=1, padding='VALID', normalizer_fn = slim.batch_norm, activation_fn=coord_act,scope='ca_conv1')
    print("y", y.shape)

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    print("x_h, x_w", x_h.shape, x_w.shape)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    print("x_w", x_w.shape)
    a_h = slim.conv2d(x_h, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv2')
    print("a_h", a_h.shape)
    a_w = slim.conv2d(x_w, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv3')
    print("a_w", a_w.shape)

    out = x * a_h * a_w
    print("out", out.shape)
    return out


# 监督对比损失函数
def SupConLoss(features, labels=None, mask=None):
    """Compute loss for model. If both `labels` and `mask` are None,
            it degenerates退化 to SimCLR unsupervised loss:
            https://arxiv.org/pdf/2002.05709.pdf

            Args:
                features: hidden vector of shape [bsz, n_views, ...].
                labels: ground truth of shape [bsz].
                mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                    has the same class as sample i. Can be asymmetric.
            Returns:
                A loss scalar.
            """
    features = tf.convert_to_tensor(features)
    labels = tf.convert_to_tensor(labels) if labels is not None else None

    temperature = 0.1
    contrast_mode = 'all'
    base_temperature = 0.07
    batch_size, n_views, C = features.get_shape().as_list()
    print(features)

    # 指定运行的GPU或CPU设备未转化

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = tf.reshape(features, [batch_size, n_views, -1])

    if features.dtype != tf.float32:
        features = tf.cast(features, tf.float32)

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = tf.eye(batch_size, dtype=tf.float32)
    elif labels is not None:
        labels = tf.reshape(labels, [-1, 1])
        print(labels.shape)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = tf.equal(labels, tf.transpose(labels))
        mask = tf.cast(mask, tf.float32)
    else:
        mask = tf.cast(mask, tf.float32)

    contrast_count = n_views
    contrast_feature = tf.reshape(
        tf.transpose(features, perm=[1, 0, 2]),
        [n_views * batch_size, -1])
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))
    # compute logits（对数）
    temperature = tf.cast(temperature, tf.float32)
    base_temperature = tf.cast(base_temperature, tf.float32)
    anchor_dot_contrast = tf.matmul(anchor_feature, tf.transpose(contrast_feature)) / temperature
    # for numerical stability
    logits = (
            anchor_dot_contrast - tf.reduce_max(tf.stop_gradient(anchor_dot_contrast), axis=1, keep_dims=True))
    # logits_max = tf.reduce_max(anchor_dot_contrast, reduction_indices=[1])
    # logits = anchor_dot_contrast - logits_max         ########

    # tile mask
    mask = tf.tile(mask, [anchor_count, contrast_count])
    # mask-out self-contrast cases

    # logits_mask = torch.scatter(tf.ones_like(mask), 1,  tf.reshape(np.arange(batch_size * anchor_count), [-1, 1]), 0)  ########
    logits_mask = tf.ones_like(mask)
    mask2 = tf.diag(tf.ones(mask.shape[0]))
    logits_mask = logits_mask - mask2
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.log(tf.reduce_sum(exp_logits, axis=1, keep_dims=True)) # 求和但不降维

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / tf.reduce_sum(mask, axis=1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]), axis=0)
    return loss



def RCAB(input, reduction):
    """
    @Image super-resolution using very deep residual channel attention networks
    Residual Channel Attention Block
    """
    batch, height, width, channel = input.get_shape()  # (B, W, H, C)
    # f = tf.layers.conv2d(input, channel, 3, padding='same', activation=tf.nn.relu)  # (B, W, H, C)  channel是输出空间的维数
    # f = tf.layers.conv2d(f, channel, 3, padding='same')  # (B, W, H, C)    以上是残差

    x = tf.reduce_mean(input, axis=(1, 2), keep_dims=True)  # (B, 1, 1, C)      以下是CA
    x = tf.layers.conv2d(x, channel // reduction, 1, activation=tf.nn.relu)  # (B, 1, 1, C // r)
    x = tf.layers.conv2d(x, channel, 1, activation=tf.nn.sigmoid)  # (B, 1, 1, C)
    x = tf.multiply(input, x)  # (B, W, H, C)

    x = tf.add(input, x)
    return x

def SRNBlock(relation_u, relation_v, scope, bn, is_training, bn_decay):
    print("relation_u", relation_u.shape)
    print("relation_v", relation_v.shape)
    batchsize, width, height, in_uchannels,  = relation_u.get_shape().as_list()
    _, _, _, in_vchannels = relation_v.get_shape().as_list()
    with tf.variable_scope(scope) as sc:

        with tf.variable_scope('gu') as scope:
            gu_output = RCAB(relation_u, 2)
            # gu_output = CoordAtt(gu_output)
            gu_output = tf_util_srn.conv2d(gu_output, 192, [1, 1], padding='VALID', stride=[1, 1],bn=bn, is_training=is_training,scope='conv1', bn_decay=bn_decay)
            print(gu_output.shape)
            # gu_output = tf.reduce_mean(gu_output, 2)
            gu_output = tf.reshape(gu_output, [batchsize, -1, 192])
            print("gu_output", gu_output.shape)

        with tf.variable_scope('gv') as scope:
            # gv_output = CoordAtt(relation_v)
            # gv_output = NonLocalBlock(gv_output, sub_sample=False, out_channels=3, is_bn=False)
            gv_output = RCAB(relation_v, 3)
            # gv_output = CoordAtt(gv_output)
            print(gv_output.shape)
            gv_output = tf_util_srn.conv2d(gv_output, 192, [1, 1], padding='VALID', stride=[1, 1],bn=bn, is_training=is_training,scope='conv2', bn_decay=bn_decay)
            # gv_output = tf.reduce_mean(gv_output, 2)
            gv_output = tf.reshape(gv_output, [batchsize, -1, 192])
            print("gv_output", gv_output.shape)

        return gu_output, gv_output

def SRNBlock1(relation_u, relation_v, scope, bn, is_training, bn_decay):
    print("relation_u", relation_u.shape)
    print("relation_v", relation_v.shape)
    batchsize, width, height, in_uchannels,  = relation_u.get_shape().as_list()
    _, _, _, in_vchannels = relation_v.get_shape().as_list()
    with tf.variable_scope(scope) as sc:

        with tf.variable_scope('gu') as scope:
            # gu_output = RCAB(relation_u, 2)
            gu_output = CoordAtt(relation_u)
            gu_output = tf_util_srn.conv2d(gu_output, 387, [1, 1], padding='VALID', stride=[1, 1],bn=bn, is_training=is_training,scope='conv10', bn_decay=bn_decay)
            print(gu_output.shape)
            gu_output = tf.reduce_mean(gu_output, 2)
            gu_output = tf.reshape(gu_output, [batchsize, -1, 387])
            print("gu_output", gu_output.shape)

        with tf.variable_scope('gv') as scope:
            # gv_output = CoordAtt(relation_v)
            # gv_output = NonLocalBlock(gv_output, sub_sample=False, out_channels=3, is_bn=False)
            # gv_output = RCAB(relation_v, 3)
            gv_output = CoordAtt(relation_v)
            print(gv_output.shape)
            gv_output = tf_util_srn.conv2d(gv_output, 387, [1, 1], padding='VALID', stride=[1, 1],bn=bn, is_training=is_training,scope='conv20', bn_decay=bn_decay)
            gv_output = tf.reduce_mean(gv_output, 2)
            gv_output = tf.reshape(gv_output, [batchsize, -1, 387])
            print("gv_output", gv_output.shape)

        return gu_output, gv_output


