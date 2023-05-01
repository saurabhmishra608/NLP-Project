import argparse
import cv2
from CNN import Classifier
from yolo import YOLO
import torch
import numpy as np
import os
import seaborn as sns
from torchvision import transforms

def videoProcess(videoPath, outputPath, cnnPath,network = "normal",device = 0,size = 416,confidence = 0.2,hands = 1):


# ap = argparse.ArgumentParser()
# ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
#                 help='Network Type')
# ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
# ap.add_argument('-s', '--size', default=416, help='Size for yolo')
# ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
# ap.add_argument('-nh', '--hands', default=1, help='Total number of hands to be detected per frame (-1 for all)')
# args = ap.parse_args()

    if network == "normal":
        print("loading yolo...")
        yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
    elif network == "prn":
        print("loading yolo-tiny-prn...")
        yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
    elif network == "v4-tiny":
        print("loading yolov4-tiny-prn...")
        yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
    else:
        print("loading yolo-tiny...")
        yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

    yolo.size = int(size)
    yolo.confidence = float(confidence)

    #print("starting webcam...")
    #cv2.namedWindow("preview")
    #vc = cv2.VideoCapture(args.device)

    print("Processing video...")
    #video_path = "/home/saurabh/dig_path/RISE_CAM16/yolo-hand-detection-master/input.mp4"
    video_path = videoPath
    #cv2.namedWindow("preview")
    vc = cv2.VideoCapture(video_path)

    # Initialize VideoWriter
    print("starting videowriter")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #video_output = '/home/saurabh/dig_path/RISE_CAM16/yolo-hand-detection-master/output_video.avi'
    video_output = outputPath
    out = cv2.VideoWriter(video_output, fourcc, 20.0, (int(vc.get(3)), int(vc.get(4))))
    length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print( "len = ",length )


    model=Classifier()
    model=model.to("cuda")
    #model.load_state_dict(torch.load('models/sign_detect.tar')['state_dict'])
    #model.load_state_dict(torch.load('/home/saurabh/dig_path/RISE_CAM16/yolo-hand-detection-master/models/hand_rec2.pt'))
    model.load_state_dict(torch.load(cnnPath))

    device = 'cuda'

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    english_letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    a = True
    frame_count = 0
    t = 0
    W  = vc.get(3)   # float `width`
    H = vc.get(4)  # float `height`
    pred_letter = ""
    text = ""
    i = 0
    while vc.isOpened():
        t = t+1
        rval, frame = vc.read()
        if rval:
            width, height, inference_time, results = yolo.inference(frame)
        else:
            break

        # display fps
        cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)
        cv2.putText(frame, pred_letter, (15, 80), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255), 2)
        cv2.putText(frame, text, (15, 150), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,255), 2)
        # sort by confidence
        results.sort(key=lambda x: x[2])

        # how many hands should be shown
        hand_count = len(results)
        if (hand_count==0):
            a = True
        if hands != -1:
            hand_count = int(hands)
        
        #display hands
        for detection in results[:hand_count]:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            
            if (a):
                frame_count = frame_count+1
                if(frame_count>20):
                    a = False
                    frame_count = 0
                    i = i+1
                    #cv2.imshow("cropped",frame[y-int(0.1*h):y+int(1.1*h),x-int(0.1*w):x+int(1.1*w)])
                    h1 = int(max(0,y-int(0.5*h)))
                    h2 = int(min(H,y+int(1.5*h)))
                    w1 = int(max(0,x-int(0.5*w)))
                    w2 = int(min(W,x+int(1.5*w)))
                    #print(h1,h2,w1,w2)
                    image = frame[h1:h2,w1:w2]
                    cv2.imwrite(os.path.join('/home/saurabh/dig_path/RISE_CAM16/yolo-hand-detection-master/images',str(i)+'.jpg'),image)
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    img= np.asarray(img)
                    #print(img)
                    img=cv2.resize(img,(28,28))
                    tensor_image=torch.FloatTensor(img)
                    #print(tensor_image.shape)
                    
                    tensor_image = tensor_image.unsqueeze(0)
                    tensor_image = tensor_image.unsqueeze(0)
                    tensor_image/=255
                    tensor_image = transforms.Normalize((0.5,), (0.5,))(tensor_image)
                    #tensor_image/=255
                    #img/=255
                    outputs=model(tensor_image.to(device))
                    #preds = model.predict(img)
                    predicted = torch.softmax(outputs,dim=1)
                    #print(predicted.shape)
                    predicted=torch.argmax(predicted, dim = 1)
                    # predicted_class = np.argmax(preds, axis=1)
                    

                    
                    
                    pred_letter = english_letters[predicted]
                    text = text+pred_letter
                    print("predicted letter = ",english_letters[predicted],"text extracted = ",text)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
            #text = "%s (%s)" % (name, round(confidence, 2))
            # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, color, 2)
            
        
            # Save the processed frame to the output video file
        out.write(frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    out.release()
    vc.release()
    return text


