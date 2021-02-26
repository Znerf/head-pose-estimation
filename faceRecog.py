# from modules.emotion import Emotion
from module.yolo import YoloDark

import cv2
from PIL import Image, ImageOps

def main():
    yolo= YoloDark()

    # model= emotion.load_weights()
    net= yolo.yoloInit()

    cap = cv2.VideoCapture(yolo.port)
    i=0
    while True:
        has_frame, frame = cap.read()
        if(i==10):
            if not has_frame:
                print('Done processing')
                cv2.waitKey(1000)
                break
            faces= yolo.yoloProcess(net,frame)

            if faces is not None:
                for face in faces:
                    if face is not None:
                        x1=face[0]
                        y1=face[1]
                        x2=face[0]+face[2]
                        y2=face[1]+face[3]
                        face = frame[y1:y2, x1:x2]
                        # image = Image.fromarray(face).convert('L')
                        # image = image.resize((48,48))


            cv2.imshow("Cam", frame)
            i=0
        i=i+1

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
