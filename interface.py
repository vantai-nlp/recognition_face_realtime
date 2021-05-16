from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time
from preprocess import preprocesses
from classifier import training
from recognition_face import recognition_face_video, recognition_face_image, recognition_face_camera
import cv2
import os


class InterFace(object):
    def __init__(self):
        self.input_datadir = './train_img'
        self.output_datadir = './pre_img'
        self.datadir = './pre_img'
        self.modeldir = './assets/20170511-185253.pb'
        self.classifier_filename = './assets/classifier.pkl'
        self.pretrain_mtcnn_dir = './assets/npy/'

    def _preprocesses(self):
        obj = preprocesses(self.pretrain_mtcnn_dir, self.input_datadir, self.output_datadir)
        nrof_images_total, nrof_successfully_aligned = obj.collect_data()
        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    def _training(self):
        print ("Training Start")
        obj=training(self.output_datadir, self.modeldir, self.classifier_filename)
        get_file=obj.main_train()
        print('Saved classifier model to file "%s"' % get_file)

    def _video2images(self):
        cap = cv2.VideoCapture(0)
        count = 0
        name_class = input("What's your name: ")
        cmd = 'cd ' + self.input_datadir + '&& mkdir ' + name_class
        os.system(cmd)
        input('Press to capture...')
        while True:

            path = os.path.join(self.input_datadir, name_class, str(count) + '.jpg')
            _, frame = cap.read()
            cv2.imshow('video', frame)
            cv2.imwrite(path, frame)
            count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q') or count == 150:
                print('Stop capture...')
                break

        cap.release()
        cv2.destroyAllWindows()


    def add_face_from_camera(self):
        self._video2images()
        self._preprocesses()
        self._training()

    def add_face_from_handcraft(self):
        self._preprocesses()
        self._training()

    def recognition_face_camera(self):
        recognition_face_camera()

    def recognition_face_video(self, video_path):
        recognition_face_video(video_path)

    def recognition_face_image(self, img_path):
        recognition_face_image(img_path)






