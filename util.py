#-- coding: utf-8 --
import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors_pre(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    return anchors
    pass


def create_anchors_tensor(height, width, feat_stride=16, scales=[8, 16, 32], ratios=[2.0, 1.0 , 0.5]):
    anchors_wh = []
    for r in ratios:
        for s in scales:
            area = np.square(16*s)
            h = np.round(np.sqrt(area/r))
            w = np.round(r * h)
            anchors_wh.append([w,h])
            pass
        pass
    anchors_wh = np.array(anchors_wh)

    grid_x = np.arange(0,width) * feat_stride + feat_stride / 2
    grid_y = np.arange(0,height) * feat_stride + feat_stride / 2
    offset_x, offset_y = np.meshgrid(grid_x, grid_y)
    offset_x = np.reshape(offset_x, (-1, 1))
    offset_y = np.reshape(offset_y, (-1, 1))

    offset_xy = np.concatenate([offset_x, offset_y], -1)
    offset_xy = np.tile(offset_xy,(1,9))
    offset_xy = np.reshape(offset_xy,(-1,9,2))
    anchors_wh = np.tile(anchors_wh,(height * width,1))
    anchors_wh = np.reshape(anchors_wh, (-1, 9, 2))
    anchors_xywh = np.concatenate([offset_xy,anchors_wh],-1)
    anchors_xywh = np.expand_dims(anchors_xywh,0)
    anchors_tensor = np.tile(anchors_xywh,(1,1,1,1))
    anchors_tensor = np.reshape(anchors_tensor,(height * width * 9,4))
    anchors_x1 = anchors_tensor[:,0] - np.round(0.5 * anchors_tensor[:,2])
    anchors_y1 = anchors_tensor[:, 1] - np.round(0.5 * anchors_tensor[:, 3])
    anchors_x2 = anchors_tensor[:, 0] + np.round(0.5 * anchors_tensor[:, 2])
    anchors_y2 = anchors_tensor[:, 1] + np.round(0.5 * anchors_tensor[:, 3])

    anchors_tensor = np.concatenate([anchors_x1[:,np.newaxis],anchors_y1[:,np.newaxis],anchors_x2[:,np.newaxis],anchors_y2[:,np.newaxis]],axis=-1)

    return anchors_tensor
    pass
if __name__ == '__main__':
    anchors_1 = create_anchors_tensor(38,57)
    anchors_2 = generate_anchors_pre(38,57)
    print(123)
    pass