# class preprocesses: load dataset and align (get faces and resize)
# algorithm: pretrain model mtcnn (pretrain file in ./npy) and use detect_face.py to load model
# input_datadir: dir dataset, output_datadir: dir contain align image (faces and resize) of each class

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import os
import tensorflow.compat.v1 as tf
import numpy as np
import facenet
import detect_face

class preprocesses:
    def __init__(self, pretrain_mtcnn_dir, input_datadir, output_datadir):
        self.input_datadir = input_datadir
        self.output_datadir = output_datadir
        self.pretrain_mtcnn_dir = pretrain_mtcnn_dir

    def collect_data(self):
        # check output_datadir, if exist -> pass and else -> create output_datadir
        output_dir = os.path.expanduser(self.output_datadir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # input: input_datadir, output: return list of class and image paths
        dataset = facenet.get_dataset(self.input_datadir)

        # load mtcnn
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, self.pretrain_mtcnn_dir)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        image_size = 182

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            
            # traverse each class in dataset
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)

                # load image and alige
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    print("Image: %s" % image_path)
                    if not os.path.exists(output_filename):
                        try:
                            img = cv2.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                                print('to_rgb data dimension: ', img.ndim)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                                        factor)
                            nrof_faces = bounding_boxes.shape[0]
                            print('No of Detected Face: %d' % nrof_faces)
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det = det[index, :]
                                det = np.squeeze(det)
                                bb_temp = np.zeros(4, dtype=np.int32)

                                bb_temp[0] = det[0]
                                bb_temp[1] = det[1]
                                bb_temp[2] = det[2]
                                bb_temp[3] = det[3]

                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                scaled_temp = cv2.resize(cropped_temp, (image_size, image_size), interpolation = cv2.INTER_LINEAR)
                                nrof_successfully_aligned += 1
                                cv2.imwrite(output_filename, scaled_temp)
                                text_file.write('%s %d %d %d %d\n' % (
                                output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))

        return (nrof_images_total,nrof_successfully_aligned)