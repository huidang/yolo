from __future__ import print_function

import logging as log
import os
import pathlib
import json
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

device = 'CPU'
input_h, input_w, input_c, input_n = (480, 800, 3, 1)
log.basicConfig(level=log.DEBUG)

#labels
label_id_map = {
    1: "a",
    2: "b",
    3: "c"
}
ie = None
exec_net = None

def init():
    """Initialize model

    Returns: model

    """
    model_xml = "/usr/local/ev_sdk/model/openvino/yolov3.xml"
    if not os.path.isfile(model_xml):
        log.error(f'{model_xml} does not exist')
        return None
    model_bin = pathlib.Path(model_xml).with_suffix('.bin').as_posix()
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # Load Inference Engine
    log.info('Loading Inference Engine')
    global ie
    ie = IECore()
    global exec_net
    exec_net = ie.load_network(network=net, device_name=device)
    log.info('Device info:')
    versions = ie.get_versions(device)
    print("{}".format(device))
    print("MKLDNNPlugin version ......... {}.{}".format(versions[device].major, versions[device].minor))
    print("Build ........... {}".format(versions[device].build_number))

    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    global input_h, input_w, input_c, input_n
    input_h, input_w, input_c, input_n = h, w, c, n

    return net

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    #box2 = box2.T
    print(box2.shape)
    print(box1.shape)
    box1 = box1[None].repeat(box2.shape[0], axis=0)
    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    
    inter_area = (np.min(np.concatenate((b1_x2[..., None], b2_x2[...,None]),1), 1) - np.max(np.concatenate((b1_x1[...,None], b2_x1[...,None]), 1), 1)).clip(0) * \
                 (np.min(np.concatenate((b1_y2[..., None], b2_y2[..., None]),1), 1) - np.max(np.concatenate((b1_y1[...,None], b2_y1[..., None]), 1), 1)).clip(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou

    return iou

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Multiply conf by class conf to get combined confidence
        class_conf= pred[:, 5:].max(1)
        class_pred = pred[:, 5:].argmax(1)
        print(f"{class_conf.shape}, {class_pred.shape}")
        pred[:, 4] *= class_conf
        print(f"1: {pred[:, 4]}")

        # # Merge classes (optional)
        # class_pred[(class_pred.view(-1,1) == torch.LongTensor([2, 3, 5, 6, 7]).view(1,-1)).any(1)] = 2
        #
        # # Remove classes (optional)
        # pred[class_pred != 2, 4] = 0.0

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres)  & np.isfinite(pred).all(1)
        print(i[i==True])
        pred = pred[i]
        print(pred.shape)

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i][..., None].astype(np.float32)

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        #pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = np.concatenate((pred[:, :5], class_conf[..., None], class_pred), 1)
        print(pred.shape)

        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        print(pred.shape)
        for c in np.unique(pred[:, -1]):
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    # dc = dc[dc[:, 4] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = np.concatenate(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


def process_image(net, input_image, thresh=0.5):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        thresh: thresh value

    Returns: process result

    """
    imgs = input_image.copy()
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
    log.info(f'process_image, ({input_image.shape}')
    ih, iw, _ = input_image.shape

    # --------------------------- Prepare input blobs -----------------------------------------------------
    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
    input_image = input_image.transpose((2, 0, 1))/255.0
    #input_image = np.ones((3, 480, 800)).astype(np.float32)*120/255.0
    images = np.ndarray(shape=(input_n, input_c, input_h, input_w))
    images[0] = input_image

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

    output_name = [out_blob]
    output_info = []
    try:
        for name in output_name:
            output_info.append(net.outputs[name])
            print(output_info[-1].shape)
    except KeyError:
        log.error(f"Can't find a {output_name} layer in the topology")
        return None

    #output_dims = output_info.shape
    #if len(output_dims) != 4:
    #    log.error("Incorrect output dimensions for SSD model")
    #max_proposal_count, object_size = output_dims[1], output_dims[2]

    if output_info[0].shape[-1] != 7:
        log.error("Output item should have 8 as a last dimension")
    for data in output_info:
        data.precision = "FP32"


    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Creating infer request and starting inference")
    res = exec_net.infer(inputs={input_blob: images})

    # --------------------------- Read and postprocess output ---------------------------------------------
    log.info("Processing output blobs")
    data = res[out_blob]
    
    detect_obj = []
    print(data.shape)
    #data[..., 5:] = data[..., 5:]*data[...,4][...,None]
    data[..., 0] = data[..., 0] - data[..., 2]/2
    data[..., 1] = data[..., 1] -data[..., 3]/2
    data[..., 2] = data[..., 0] + data[..., 2]
    data[..., 3] = data[..., 1] + data[..., 3] 
    output = non_max_suppression(data)
    for box in output[0]:
        box[0] = box[0] / 800 * iw
        box[1] = box[1] / 800 * ih
        box[2] = box[2] /800 * iw
        box[3] = box[3] /800 *ih
        print(f"{box[0]}, {box[1]}, {box[2]}, {box[3]}")
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        #print(f"{xmin}, {ymin}, {xmax}, {ymax}")
        label = int(np.argmax(box[5:])) + 1
        if label not in label_id_map:
            continue
        if box[5:][label-1] > thresh:
            detect_obj.append(
                {'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'name': label_id_map[label]}
            )
        cv2.rectangle(imgs, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        #cv2.rectangle(imgs, (0, 0), (100, 100), 255, 2)
    cv2.imwrite('output.jpg', imgs)    
    #tmp = {"objects": detect_obj}
    out = '{"objects":'+json.dumps(detect_obj)+"}"
    return out


if __name__ == '__main__':
    # Test API
    img = cv2.imread('/usr/local/ev_sdk/src/tmp/MotorHTL20200610_162.jpg')
    predictor = init()
    print("init done.")
    result = process_image(predictor, img, 0.5)
    #log.info(result)
