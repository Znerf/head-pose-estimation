"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""

from multiprocessing import Process, Queue

import cv2
import csv
import numpy as np

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

from module.yolo import YoloDark
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

print("OpenCV version: {}".format(cv2.__version__))

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128


def get_face(detector,img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    yolo= YoloDark()
    net= yolo.yoloInit()
    while True:
        image = img_queue.get()
        box = yolo.faces(image)#detector.extract_cnn_facebox(image)
        box_queue.put(box)


def main():
    # with open('oldface.csv') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=' ')
    #     line_count = 0
    #     datarec=[]
    #     for row in csv_reader:
    #         if line_count == 0:
                
    #             line_count += 1
    #         else:
    #             print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
    #             datarec.append(row)
    #             line_count += 1
    #     print(datarec)
    video_src = 0#"data/sample.mp4"#""0
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    # img_queue = Queue()
    # box_queue = Queue()
    # img_queue.put(sample_frame)
    # box_process = Process(target=get_face, args=(mark_detector,img_queue, box_queue))
    # box_process.start()

    
    yolo= YoloDark()
    net= yolo.yoloInit()
    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()
    side=[]
    up=[]

    while True:
        frame_got, frame = cap.read()
        if frame_got is False:
            break
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        if cv2.waitKey(10) == 27:
            break
        facebox1 = yolo.faces(frame)
        # print(box)
        # continue
        # img_queue.put(frame)

        # Get face from box queue.
        # facebox1 = box_queue.get()
        
        for facebox in facebox1:
            if facebox is None:
                continue
            # print(facebox)
            # Detect landmarks from image of 128x128.
            try:
                # face_img = frame[facebox[1]: facebox[3],
                #                 facebox[0]: facebox[2]]
                if (facebox[0]-10)>0:
                    facebox[0]=facebox[0]-15
                if (facebox[1]-10)>0:
                    facebox[1]=facebox[1]-15
                # if (facebox[1]+facebox[2]+50)<shape[1]:
                facebox[2]=facebox[2]+60
                # if (facebox[3]+facebox[3]+0)<shape[0]:
                facebox[3]=facebox[3]+30

                x1=facebox[0]
                # if (x1-5)>0:
                #     x1=x1-50
                    
                # if x1<0:
                #     x1=0
                y1=facebox[1]
                # if (y1-5)>0:
                #     y1=y1-50
                # if x1<0:
                #     y1=0
                x2=facebox[0]+facebox[2]
                # if (x2+5)<frame.get(3):
                # if x2> vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH):
                #     x2= vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
                y2=facebox[1]+facebox[3]
                # if ((y2+5)<frame.get(4)):
                # if y2> vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT):
                #     y2= vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
                face_img = frame[y1:y2, x1:x2]
                # box = detector.extract_cnn_facebox(face_img)
                # if box is not None:
                #     # Detect landmarks from image of 128x128.
                #     face_img = frame[box[1]: box[3],
                #                     box[0]: box[2]]
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                # cv2.imshow("Preview1", face_img)
                tm.start()
                marks = mark_detector.detect_marks([face_img])
                tm.stop()
                # cv2.imshow("Preview1", face)
                # Convert the marks locations from local CNN to global image.
                marks *= (facebox[2])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]

                # Uncomment following line to show raw marks.
                mark_detector.draw_marks(
                    frame, marks, color=(0, 255, 0))

                # Uncomment following line to show facebox.
                # mark_detector.draw_box(frame, [facebox])

                # Try pose estimation with 68 points.
                pose = pose_estimator.solve_pose_by_68_points(marks)

                # Stabilize the pose.
                # steady_pose = []
                # pose_np = np.array(pose).flatten()
                # for value, ps_stb in zip(pose_np, pose_stabilizers):
                #     ps_stb.update([value])
                #     steady_pose.append(ps_stb.state[0])
                # steady_pose = np.reshape(steady_pose, (-1, 3))
                # pose_np = np.array(pose).flatten()
                # pose = np.reshape(pose, (-1, 3))

                # Uncomment following line to draw pose annotation on frame.
                # print (pose[1])
                # print(pose[1][2]/3.14)
                pose_estimator.draw_annotation_box(
                    frame, pose[0], pose[1], color=(255, 128, 128))

                # Uncomment following line to draw stabile pose annotation on frame.
                # pose_estimator.draw_annotation_box(
                #     frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))
                # break

                # Uncomment following line to draw head axes on frame.
                # print(ste)
                pose_estimator.draw_axes(frame, pose[0], pose[1])
                ratio =pose_estimator.cal_eye_index(marks)

                side.append((ratio[0]-0.2)/(0.8-0.2)*100-50)
                up.append((pose[1][1]+60)/(100+60)*100-50)
                print(pose[1][1])
                # up.append(pose[1][1])
                # print((pose[1][1]+60)/(-200+60)*100)
                print(pose[1][1])

                # image_points=marks

                # # (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)

                # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), pose[1], pose[0], pose_estimator.ret()[1],pose_estimator.ret()[0])
            #     vec,
            # tvec=self.t_vec[0]), int(nose_end_point2D[0][0][1]))
                # print('sdf')
                # cv2.line(frame, p1, p2, (255,0,0), 2)
            except:
                print("NONE")
        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            plt.scatter(side, up, s=np.pi*3, c=(0,0,0), alpha=0.5)
            plt.title('Scatter plot')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
            break
        
    # Clean up the multiprocessing process.
    # box_process.terminate()
    # box_process.join()
    



if __name__ == '__main__':
    main()
