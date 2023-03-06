import cv2
import sys
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd_predictor

def get_model(label_path="Transfer-Learing-SSD-model/data/open-images-model-labels.txt", model_path = "Transfer-Learing-SSD-model/models/ssd-mb1-100-120-Loss-2.7290525138378143.pth"):
    class_names = [name.strip() for name in open(label_path).readlines()] 
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)          
    net.load(model_path)                                                  
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=100) 

    return predictor, class_names

def detect_object_in_image(image_path, label_path="Transfer-Learing-SSD-model/data/open-images-model-labels.txt",percent=0.6):
    exclude_name=image_path.split("/")[-1][:-4]
    output_path=f"Transfer-Learing-SSD-model/output/result_{exclude_name}.jpg"
    predictor, class_names = get_model(label_path)

    origin_image = cv2.imread(image_path)
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, percent)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(origin_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(origin_image, label, (int(box[0]) + 30, int(box[1]) + 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imwrite(output_path, origin_image)

def detect_object_in_video(video_path, label_path="Transfer-Learing-SSD-model/data/open-images-model-labels.txt", model_path = "Transfer-Learing-SSD-model/models/ssd-mb1-100-120-Loss-2.7290525138378143.pth"):
    
    videoWriter = None
    videoCapture = cv2.VideoCapture(video_path)
    predictor, class_names = get_model(label_path, model_path)
    exclude_name=video_path.split("/")[-1][:-4]
    output_path=f"Transfer-Learing-SSD-model/output/result_{exclude_name}.avi"


    while True:
        ret, origin_image = videoCapture.read()
        
        if not ret:
            break
        
        if origin_image is None:
            continue
        
        if videoWriter is None:
            height, width, layers = origin_image.shape
            videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
        
        image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, 10, 0.6)
        
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(origin_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
            cv2.putText(origin_image, label, (int(box[0]) + 20, int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        
        videoWriter.write(origin_image)
    
    videoWriter.release()
    videoCapture.release()
    cv2.destroyAllWindows()



video_path='Transfer-Learing-SSD-model/vid5.mp4'
detect_object_in_video(video_path)
