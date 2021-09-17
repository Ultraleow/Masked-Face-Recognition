import os,math,cv2,shutil
import numpy as np
import tensorflow

if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile

print("Tensorflow version: ",tf.__version__)




img_format = {'png','jpg','bmp'}


def model_restore_from_pb(pb_path,node_dict,GPU_ratio=None):
    tf_dict = dict()
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,
                                )
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio
        sess = tf.Session(config=config)
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            #----issue solution if models with batch norm
            '''
            if batch normalzition：
            ValueError: Input 0 of node InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/cond_1/AssignMovingAvg/Switch was passed 
            float from InceptionResnetV2/Conv2d_1a_3x3/BatchNorm/moving_mean:0 incompatible with expected float_ref.
            ref:https://blog.csdn.net/dreamFlyWhere/article/details/83023256
            '''
            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            tf.import_graph_def(graph_def, name='')

        sess.run(tf.global_variables_initializer())
        for key,value in node_dict.items():
            node = sess.graph.get_tensor_by_name(value)
            tf_dict[key] = node
        return sess,tf_dict

def img_removal_by_embed(root_dir,output_dir,pb_path,node_dict,threshold=0.7,type='copy',GPU_ratio=None, dataset_range=None):
    # ----var
    img_format = {"png", 'jpg', 'bmp'}
    batch_size = 64

    # ----collect all folders
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No sub-dirs in ", root_dir)
    else:
        #----dataset range
        if dataset_range is not None:
            dirs = dirs[dataset_range[0]:dataset_range[1]]

        # ----model init
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
        tf_input = tf_dict['input']
        tf_phase_train = tf_dict['phase_train']
        tf_embeddings = tf_dict['embeddings']
        model_shape = [None, 160, 160, 3]
        feed_dict = {tf_phase_train: False}

        # ----tf setting for calculating distance
        with tf.Graph().as_default():
            tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
            tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
            tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
            # ----GPU setting
            config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                    )
            config.gpu_options.allow_growth = True
            sess_cal = tf.Session(config=config)
            sess_cal.run(tf.global_variables_initializer())



        #----process each folder
        for dir_path in dirs:
            paths = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            len_path = len(paths)
            if len_path == 0:
                print("No images in ",dir_path)
            else:
                # ----create the sub folder in the output folder
                save_dir = os.path.join(output_dir, dir_path.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # ----calculate embeddings
                ites = math.ceil(len_path / batch_size)
                embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
                for idx in range(ites):
                    num_start = idx * batch_size
                    num_end = np.minimum(num_start + batch_size, len_path)
                    # ----read batch data
                    batch_dim = [num_end - num_start]#[64]
                    batch_dim.extend(model_shape[1:])#[64,160, 160, 3]
                    batch_data = np.zeros(batch_dim, dtype=np.float32)
                    for idx_path,path in enumerate(paths[num_start:num_end]):
                        img = cv2.imread(path)
                        if img is None:
                            print("Read failed:",path)
                        else:
                            img = cv2.resize(img, (model_shape[2], model_shape[1]))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            batch_data[idx_path] = img
                    batch_data /= 255  # norm
                    feed_dict[tf_input] = batch_data
                    embeddings[num_start:num_end] = sess.run(tf_embeddings, feed_dict=feed_dict)

                # ----calculate ave distance of each image
                feed_dict_2 = {tf_ref: embeddings}
                ave_dis = np.zeros(embeddings.shape[0], dtype=np.float32)
                for idx, embedding in enumerate(embeddings):
                    feed_dict_2[tf_tar] = embedding
                    distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                    ave_dis[idx] = np.sum(distance) / (embeddings.shape[0] - 1)
                # ----remove or copy images
                for idx,path in enumerate(paths):
                    if ave_dis[idx] > threshold:
                        print("path:{}, ave_distance:{}".format(path,ave_dis[idx]))
                        if type == "copy":
                            save_path = os.path.join(save_dir,path.split("\\")[-1])
                            shutil.copy(path,save_path)
                        elif type == "move":
                            save_path = os.path.join(save_dir,path.split("\\")[-1])
                            shutil.move(path,save_path)

def check_path_length(root_dir,output_dir,threshold=5):
    # ----var
    img_format = {"png", 'jpg'}

    # ----collect all dirs
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]

    if len(dirs) == 0:
        print("No dirs in ",root_dir)
    else:
        # ----process each dir
        for dir_path in dirs:
            leng = len([file.name for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format])
            if leng <= threshold:
                corresponding_dir = os.path.join(output_dir,dir_path.split("\\")[-1])
                leng_corre = len([file.name for file in os.scandir(corresponding_dir) if file.name.split(".")[-1] in img_format])
                print("dir name:{}, quantity of origin:{}, quantity of removal:{}".format(dir_path.split("\\")[-1],leng,leng_corre))

def delete_dir_with_no_img(root_dir):
    # ----collect all dirs
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No dirs in ",root_dir)
    else:
        # ----process each dir
        for dir_path in dirs:
            leng = len([file.name for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format])
            if leng == 0:
                shutil.rmtree(dir_path)
                print("Deleted:",dir_path)

def random_img_select(root_dir,output_dir,select_num=1,total_num=None):
    #----var
    img_count = 0


    #----
    for obj in os.scandir(root_dir):
        if obj.is_dir():
            paths = [file.path for file in os.scandir(obj.path) if file.name[-3:] in img_format]
            len_path = len(paths)
            if len_path > 0:
                paths = np.random.choice(paths,select_num)
                for path in paths:
                    splits = path.split("\\")
                    new_path = "{}_{}".format(splits[-2],splits[-1])
                    new_path = os.path.join(output_dir,new_path)
                    shutil.copy(path,new_path)

                    img_count += 1

                #----image quantity check
                if total_num is not None:
                    if img_count >= total_num:
                        break


def face_matching_evaluation(root_dir,face_databse_dir,pb_path,test_num=1000,GPU_ratio=None):
    #----var
    paths = list()
    node_dict = {'input': 'input:0',
                 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 # 'keep_prob':'keep_prob:0'
                 }
    batch_size = 128

    #----get all images
    for dir_name, subdir_names, filenames in os.walk(root_dir):
        if len(filenames):
            for file in filenames:
                if file[-3:] in img_format:
                    paths.append(os.path.join(dir_name,file))

    if len(paths) == 0:
        print("No images in ",root_dir)
    else:
        # paths = np.random.choice(paths,test_num)
        paths = paths[:test_num]

        #----get images of face_databse_dir
        paths_ref = [file.path for file in os.scandir(face_databse_dir) if file.name[-3:] in img_format]
        len_path_ref = len(paths_ref)
        if len_path_ref == 0:
            print("No images in ", face_databse_dir)
        else:
            #----model init
            sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
            tf_embeddings = tf_dict['embeddings']


            # ----tf setting for calculating distance
            with tf.Graph().as_default():
                tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
                tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
                tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
                # ----GPU setting
                config = tf.ConfigProto(log_device_placement=True,
                                        allow_soft_placement=True,
                                        )
                config.gpu_options.allow_growth = True
                sess_cal = tf.Session(config=config)
                sess_cal.run(tf.global_variables_initializer())

            # ----calculate embeddings
            embed_ref = get_embeddings(sess,paths_ref,tf_dict,batch_size=batch_size)
            embed_tar = get_embeddings(sess,paths,tf_dict,batch_size=batch_size)
            print("embed_ref shape: ",embed_ref.shape)
            print("embed_tar shape: ",embed_tar.shape)

            #----calculate distance and get the minimum
            arg_dis = list()
            dis_list = list()
            count_o = 0
            count_unknown = 0
            feed_dict_2 = {tf_ref: embed_ref}
            for idx, embedding in enumerate(embed_tar):
                feed_dict_2[tf_tar] = embedding
                distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                arg_temp = np.argsort(distance)[:1]
                arg_dis.append(arg_temp)
                dis_list.append(distance[arg_temp])

            for path, arg_list,dises in zip(paths,arg_dis,dis_list):
                # answer = path.split("\\")[-2]
                answer = path.split("\\")[-1].split('_')[0]
                for arg,dis in zip(arg_list,dises):
                    if dis < 0.7:
                        prediction = paths_ref[arg].split("\\")[-1].split("_")[0]
                        if prediction == answer:
                            count_o += 1
                            break
                    else:
                        count_unknown += 1

            #----statistics
            print("accuracy: ",count_o /len(paths) )
            print("unknown: ",count_unknown /len(paths) )




def get_embeddings(sess,paths,tf_dict,batch_size=128):
    #----
    len_path = len(paths)
    tf_input = tf_dict['input']
    tf_phase_train = tf_dict['phase_train']
    tf_embeddings = tf_dict['embeddings']

    feed_dict = {tf_phase_train: False}
    if 'keep_prob' in tf_dict.keys():
        tf_keep_prob = tf_dict['keep_prob']
        feed_dict[tf_keep_prob] = 1.0

    # model_shape = tf_input.shape
    model_shape = [None,160,160,3]
    print("tf_input shape:",model_shape)

    #----
    ites = math.ceil(len_path / batch_size)
    embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
    for idx in range(ites):
        num_start = idx * batch_size
        num_end = np.minimum(num_start + batch_size, len_path)
        # ----read batch data
        batch_dim = [num_end - num_start]  # [64]
        batch_dim.extend(model_shape[1:])  # [64,160, 160, 3]
        batch_data = np.zeros(batch_dim, dtype=np.float32)
        for idx_path, path in enumerate(paths[num_start:num_end]):
            img = cv2.imread(path)
            if img is None:
                print("Read failed:", path)
            else:
                img = cv2.resize(img, (model_shape[2], model_shape[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data[idx_path] = img
        batch_data /= 255  # norm
        feed_dict[tf_input] = batch_data
        embeddings[num_start:num_end] = sess.run(tf_embeddings, feed_dict=feed_dict)


    return embeddings


if __name__ == "__main__":
    # root_dir = r"D:\CASIA\CASIA-WebFace_aligned"
    # output_dir = r"D:\CASIA\mislabeled"
    # pb_path = r"C:\Users\JohnnyKavnie\Desktop\20180402-114759\20180402-114759.pb"
    # node_dict = {'input': 'input:0',
    #              'phase_train': 'phase_train:0',
    #              'embeddings': 'embeddings:0',
    #              }
    # dataset_range = [0,3000]
    # img_removal_by_embed(root_dir, output_dir, pb_path, node_dict, threshold=1.25, type='move', GPU_ratio=0.25,
    #                      dataset_range=dataset_range)

    # ----check_path_length
    # root_dir = r"D:\CASIA\CASIA-WebFace_aligned"
    # output_dir = r"D:\CASIA\mislabeled"
    # check_path_length(root_dir, output_dir, threshold=3)

    #----delete_dir_with_no_img
    # root_dir = r"D:\CASIA\CASIA-WebFace_aligned"
    # delete_dir_with_no_img(root_dir)


    #----random_img_select00
    # random_img_select(root_dir, output_dir, select_num=select_num,total_num=total_num)

    #----face matching evaluation
    root_dir = r"D:\dataset\CASIA\test_database_3\with_mask"
    face_databse_dir = r"D:\dataset\CASIA\test_database_3\no_mask"
    pb_path = r"G:\我的雲端硬碟\Python\Code\model_saver\face_reg_models\FLW_0.98\pb_model.pb"
    # pb_path = r"D:\code\model_saver\FaceNet_tutorial\test_2\pb_model.pb"
    # pb_path = r"D:\code\model_saver\FaceNet_tutorial\test_3_modification_all\pb_model.pb"
    # pb_path = r"D:\code\model_saver\FaceNet_tutorial\CASIA_home_eachSelect_1\pb_model.pb"
    face_matching_evaluation(root_dir, face_databse_dir, pb_path, test_num=10000, GPU_ratio=None)

    # root_dir = r"D:\dataset\ms1m_align"
    # output_dir = r"D:\dataset\CASIA\test_database_3"
    # select_num = 1
    # total_num = 100

