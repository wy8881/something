import math
from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from ultralytics import YOLO
import os
from sort import *
from flask_socketio import SocketIO, emit
import torch

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app,debug=True,cors_allowed_origins='*')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO11 pose model
model_pose = YOLO("yolo11n-pose.pt")
model_detect = YOLO("yolo11n.pt")
DETECTED_LIST = ['laptop']
ACTIVE_KEYPOINT = [9,10]
FACE_THRESHOD = 90
CLASSIFIER_FOLDER = "classifier"
if not os.path.exists(CLASSIFIER_FOLDER):
    os.makedirs(CLASSIFIER_FOLDER)
tracked_class = ['person','laptop']

@app.route('/initial', methods=['POST'])
def setModel():
    try:
        houseId = request.form.get('houseId')
        url = request.form.get('modelUrl')
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            modelName = f"{CLASSIFIER_FOLDER}/{houseId}.yml"
            with open(modelName, 'wb') as file:
                for chunk in response.iter_content(chunk_size=100 * 1024 * 1024): 
                    file.write(chunk)
                    return jsonify({'state': "sucess"})

        return jsonify({'state': "failure"}), 400
    except Exception as e:
        print(e)
        return jsonify({'state': "failure"}), 400
    

# Route to handle multipart form data (file upload)
@app.route('/checkAlive', methods = ['GET'])
def checkAlive():
    return jsonify({'state': "sucess"}),200

@socketio.on('send_frame')
def handle_frame(data):
    try:
        # Decode the received image from bytes
        image_bytes = np.frombuffer(data['frame'], dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        houseId = data['houseId']
        timeStamp = data['timeStamp']
        result = process_image(image, houseId)
        emit('frame_processed', {'result': result, 'timeStamp': timeStamp})

    except Exception as e:
        emit('error', {'error': str(e)})
def get_person(faces, people):
    found_face = dict()
    for face in faces:
        x1,y1,w,h,identity = face
        x2=x1+w
        y2=y1+h

        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        nearest_person = -1
        nearest_distance = float('inf')
        
        for idx, person in enumerate(people):
            xp1,yp1,xp2,yp2,*_=person
            if (x1 >= xp1 and y1 >= yp1 and x2 <= xp2 and y2 <= yp2):
                person_center_x = (xp1 + xp2) / 2
                person_center_y = (yp1 + yp2) / 2

                distance = math.sqrt((face_center_x - person_center_x) ** 2 + (face_center_y - person_center_y) ** 2)
                
                if distance < nearest_distance and idx not in found_face.keys():
                    nearest_distance = distance
                    nearest_person = idx
        
        if nearest_person != -1:
            found_face[idx] = identity
    
    return found_face

def process_image(image, houseId):
    try:
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_model_file = f"{houseId}.yml"
        recognizer.read(os.path.join(CLASSIFIER_FOLDER,face_model_file))
        imH, imW = image.shape[:2]
        minW = 0.01*imW
        minH = 0.01*imH
        
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Use the YOLO11 pose model to detect poses
        results_pose = model_pose.predict(bgr_image)  
        results_object = model_detect.predict(bgr_image)
        detected_object =[]
        person_detection=[]
        object_result = results_object[0]
        boxes = object_result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            conf = box.conf[0].item()  
            class_label = model_detect.names[class_id]
            if class_label in DETECTED_LIST and conf > 0.6:
                x1, y1, x2, y2 = box.xyxy[0]
                detected_object.append([x1.item(), y1.item(), x2.item(), y2.item(), conf, class_label])
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255),1) 
                cv2.putText(image, f"{class_label}", (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        pose_result = results_pose[0]
        for box, keypoint in zip(pose_result.boxes, pose_result.keypoints):
            conf = box.conf[0].item() 
            px1, py1, px2, py2 = box.xyxy[0]
            if conf > 0.6:
                # cv2.rectangle(image, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255),1) 
                # cv2.putText(image, f"{conf}", (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 
                interaction_detected = False
                keypoints = keypoint.xy.cpu().numpy()[0]
                index = None
                for i in ACTIVE_KEYPOINT:
                        x_kp, y_kp = keypoints[i] 
                        for i, obj in enumerate(detected_object):
                            ox1, oy1, ox2, oy2, obj_conf, obj_class = obj 
                            
                            if ox1 <= x_kp <= ox2 and oy1 <= y_kp <= oy2:
                                object_bounding_box = [ox1, oy1, ox2, oy2]
                                interaction_detected = True
                                index = i
                                break  
                        
                        if interaction_detected:
                            break  
                person_detection.append([px1.item(), py1.item(), px2.item(), py2.item(), conf, index,None])
        
        face_detections = []
        frame_gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)            
        faces = haar_cascade.detectMultiScale(
            frame_gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH))
        )
        for face in faces:
            (x,y,w,h) = face
            identity, confidence = recognizer.predict(frame_gray[y:y+h, x:x+w])
            if (confidence < FACE_THRESHOD):
                identity = identity
            else:
                identity = None
            
            detection = [float(x),float(y),float(w),float(h), float(identity)]
            face_detections.append(detection)
        found_face = get_person(face_detections, person_detection)
        for idx, identity in found_face.items():
            person_detection[idx][6] = identity

        # Extract keypoints from results (x, y coordinates for each joint, confidence score)
        detection_results = []
        # detection_results["object"] = detected_object
        # detection_results["people"] = person_detection
        # detection_results["face"] = face_detections
        detection_results.append(detected_object)
        detection_results.append(person_detection)
       
    except Exception as e:
        print(f"error: {e}")
    return detection_results 


if __name__ == '__main__':
   socketio.run(app, host='0.0.0.0', port=5000, debug=True)