import numpy as np
import tensorflow

import os, time, cv2
import matplotlib.pyplot as plt

if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile

print("Tensorflow version: ",tf.__version__)

def model_restore_from_pb(pb_path, node_dict,GPU_ratio=None):
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,  #print out GPU or CPU is adopted
                                allow_soft_placement=True,  #allow tf to use alternative devices
                                )
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True  # The program can access as much resource as possible
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio  # limit the GPU resource
        sess = tf.Session(config=config)
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # import the calculation graph
        sess.run(tf.global_variables_initializer())
        for key, value in node_dict.items():
            node = sess.graph.get_tensor_by_name(value)
            node_dict[key] = node
        return sess, node_dict

class FaceMaskDetection():
    def __init__(self,pb_path,margin=44,GPU_ratio=0.1):
        # ----var
        node_dict = {'input': 'data_1:0',
                     'detection_bboxes': 'loc_branch_concat_1/concat:0',
                     'detection_scores': 'cls_branch_concat_1/concat:0'}
        conf_thresh = 0.8
        iou_thresh = 0.7

        # ====anchors config
        feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5
        id2class = {0: 'Mask', 1: 'NoMask'}

        # ----model init
        # ====generate anchors
        anchors = self.generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        anchors_exp = np.expand_dims(anchors, axis=0)

        # ====model restore from pb file
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict,GPU_ratio = GPU_ratio)
        tf_input = tf_dict['input']
        model_shape = tf_input.shape  # [N,H,W,C]
        print("model_shape = ", model_shape)
        img_size = (int(tf_input.shape[2]),int(tf_input.shape[1]))
        detection_bboxes = tf_dict['detection_bboxes']
        detection_scores = tf_dict['detection_scores']

        # ----local var to global
        self.model_shape = model_shape
        self.img_size = img_size
        self.sess = sess
        self.tf_input = tf_input
        self.detection_bboxes = detection_bboxes
        self.detection_scores = detection_scores
        self.anchors_exp = anchors_exp
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.id2class = id2class
        self.margin = margin

    def generate_anchors(self,feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
        '''
        generate anchors.
        :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
        :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
        :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
        :param offset: default to 0.5
        :return:
        '''
        anchor_bboxes = []
        for idx, feature_size in enumerate(feature_map_sizes):
            cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
            cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
            cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
            center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

            num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
            center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
            anchor_width_heights = []

            # different scales with the first aspect ratio
            for scale in anchor_sizes[idx]:
                ratio = anchor_ratios[idx][0]  # select the first ratio
                width = scale * np.sqrt(ratio)
                height = scale / np.sqrt(ratio)
                anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

            # the first scale, with different aspect ratios (except the first one)
            for ratio in anchor_ratios[idx][1:]:
                s1 = anchor_sizes[idx][0]  # select the first scale
                width = s1 * np.sqrt(ratio)
                height = s1 / np.sqrt(ratio)
                anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

            bbox_coords = center_tiled + np.array(anchor_width_heights)
            bbox_coords_reshape = bbox_coords.reshape((-1, 4))
            anchor_bboxes.append(bbox_coords_reshape)
        anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
        return anchor_bboxes

    def decode_bbox(self,anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
        '''
        Decode the actual bbox according to the anchors.
        the anchor value order is:[xmin,ymin, xmax, ymax]
        :param anchors: numpy array with shape [batch, num_anchors, 4]
        :param raw_outputs: numpy array with the same shape with anchors
        :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
        :return:
        '''
        anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
        anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
        anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
        anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
        raw_outputs_rescale = raw_outputs * np.array(variances)
        predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
        predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
        predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
        predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
        predict_xmin = predict_center_x - predict_w / 2
        predict_ymin = predict_center_y - predict_h / 2
        predict_xmax = predict_center_x + predict_w / 2
        predict_ymax = predict_center_y + predict_h / 2
        predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
        return predict_bbox

    def single_class_non_max_suppression(self,bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
        '''
        do nms on single class.
        Hint: for the specific class, given the bbox and its confidence,
        1) sort the bbox according to the confidence from top to down, we call this a set
        2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
        3) remove the bbox whose IOU is higher than the iou_thresh from the set,
        4) loop step 2 and 3, util the set is empty.
        :param bboxes: numpy array of 2D, [num_bboxes, 4]
        :param confidences: numpy array of 1D. [num_bboxes]
        :param conf_thresh:
        :param iou_thresh:
        :param keep_top_k:
        :return:
        '''
        if len(bboxes) == 0: return []

        conf_keep_idx = np.where(confidences > conf_thresh)[0]

        bboxes = bboxes[conf_keep_idx]
        confidences = confidences[conf_keep_idx]

        pick = []
        xmin = bboxes[:, 0]
        ymin = bboxes[:, 1]
        xmax = bboxes[:, 2]
        ymax = bboxes[:, 3]

        area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
        idxs = np.argsort(confidences)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # keep top k
            if keep_top_k != -1:
                if len(pick) >= keep_top_k:
                    break

            overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
            overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
            overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
            overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
            overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
            overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
            overlap_area = overlap_w * overlap_h
            overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

            need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
            idxs = np.delete(idxs, need_to_be_deleted_idx)

        # if the number of final bboxes is less than keep_top_k, we need to pad it.
        # TODO
        return conf_keep_idx[pick]

    def inference(self,img_4d,ori_height,ori_width):
        # ----var
        re_boxes = list()
        re_confidence = list()
        re_classes = list()
        re_mask_id = list()

        y_bboxes_output, y_cls_output = self.sess.run([self.detection_bboxes, self.detection_scores],
                                                      feed_dict={self.tf_input: img_4d})
        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = self.decode_bbox(self.anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = self.single_class_non_max_suppression(y_bboxes, bbox_max_scores,  conf_thresh=self.conf_thresh,
                                                          iou_thresh=self.iou_thresh )
        # ====draw bounding box
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            #print("conf = ",conf)
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            #print(bbox)

            xmin = np.maximum(0, int(bbox[0] * ori_width - self.margin / 2))
            ymin = np.maximum(0, int(bbox[1] * ori_height - self.margin / 2))
            xmax = np.minimum(int(bbox[2] * ori_width + self.margin / 2), ori_width)
            ymax = np.minimum(int(bbox[3] * ori_height + self.margin / 2), ori_height)

            re_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            re_confidence.append(conf)
            re_classes.append('face')
            re_mask_id.append(class_id)
        return re_boxes, re_confidence, re_classes, re_mask_id

def img_alignment(root_dir,output_dir,margin=44,GPU_ratio=0.1,img_show=False,dataset_range=None):
    # ----record the start time
    d_t = time.time()
    # ----var
    face_mask_model_path = r'face_mask_detection.pb'
    img_format = {'png','bmp','jpg'}
    width_threshold = 60 + margin // 2
    height_threshold = 70 + margin // 2
    quantity = 0

    # ----collect all folders
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No sub folders in ",root_dir)
    else:
        dirs.sort()
        print("Total class number: ", len(dirs))
        if dataset_range is not None:
            dirs = dirs[dataset_range[0]:dataset_range[1]]
            print("Working classes: {} to {}".format(dataset_range[0], dataset_range[1]))
        else:
            print("Working classes:All")

        # ----init of face detection model
        fmd = FaceMaskDetection(face_mask_model_path, margin, GPU_ratio)

        # ----handle images of each dir
        for dir_path in dirs:
            paths = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            if len(paths) == 0:
                print("No images in ", dir_path)
            else:
                # ----create the save dir
                save_dir = os.path.join(output_dir, dir_path.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                #----read images
                quantity += len(paths)
                for idx, path in enumerate(paths):
                    img = cv2.imread(path)
                    if img is None:
                        print("Read failed:", path)
                    else:
                        ori_height, ori_width = img.shape[:2]
                        img_ori = img.copy()
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, fmd.img_size)
                        img = img.astype(np.float32)
                        img /= 255
                        img_4d = np.expand_dims(img, axis=0)
                        bboxes, re_confidence, re_classes, re_mask_id = fmd.inference(img_4d, ori_height, ori_width)
                        if len(bboxes) == 0:
                            print("No face detected on:",path)
                        else:
                            #----find out the biggest product
                            product = np.array(bboxes)
                            #print(product.shape)
                            product = product[:,-2:]
                            product = product[:,0] * product[:,1]
                            argmax = np.argmax(product)

                            filename = path.split("\\")[-1]
                            for num, bbox in enumerate(bboxes):
                                if bbox[2] >= width_threshold and bbox[3] >= height_threshold and num == argmax:
                                # if num == argmax:
                                    img_crop = img_ori[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
                                    # save_path = os.path.join(save_dir, str(idx) + '_' + str(num) + ".png")
                                    if num == 0:
                                        save_path = os.path.join(save_dir, path.split("\\")[-1])
                                    else:
                                        save_path = "{}_{}.{}".format(filename.split(".")[0],str(num),filename.split(".")[-1])
                                        save_path = os.path.join(save_dir, save_path)
                                    cv2.imwrite(save_path, img_crop)

                                    # ----display images
                                    if img_show is True:
                                        plt.subplot(1,2,1)
                                        plt.imshow(img_ori[:,:,::-1])

                                        plt.subplot(1, 2, 2)
                                        plt.imshow(img_crop[:, :, ::-1])

                                        plt.show()

                                else:
                                    print("under threshold: w={},h={},path:{}".format(bbox[2],bbox[3],path))




    # ----statistics(to know the average process time of each image)
    if quantity != 0:
        d_t = time.time() - d_t
        print("ave process time of each image:", d_t / quantity)

if __name__ == "__main__":
    #----alignment
    root_dir = r"D:\lfw\ori"
    output_dir = r"D:\lfw\aligned(False)"
    margin = 20
    GPU_ratio = None
    img_show = False
    dataset_range = None
    img_alignment(root_dir, output_dir, margin=margin, GPU_ratio=GPU_ratio, img_show=img_show,dataset_range=dataset_range)