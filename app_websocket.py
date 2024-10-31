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
from websocket_utils import *
from multiprocessing import Queue
import threading
houses = defaultdict(House)
task_queue = Queue(maxsize=1000)

from awscrt import mqtt, http
from awsiot import mqtt_connection_builder
from utils.command_line_utils import CommandLineUtils
CWD_PATH = os.getcwd()
CONSTANT_FILE = "constants.json"
WD_PATH = os.getcwd()
CONFIG_FILE = "config.json"
CONSTANT_FILE = "constants.json"
STORAGE_FILE = "storage.json"
STORAGE_PATH = CWD_PATH + "/" + STORAGE_FILE
with open(CWD_PATH + "/" + CONSTANT_FILE, 'r') as f:
    constants = json.load(f)
QOS = getattr(mqtt.QoS, constants["QOS"])
STOP = constants["STOP"]
CONNECTION = constants["CONNECTION"]
CONNECTION_FAILURE = constants["CONNECTION_FAILURE"]
CONNECTION_SUCCESS = constants["CONNECTION_SUCCESS"]
CONNECTION_RESUMED = constants["CONNECTION_RESUMED"]
RECEIVE = constants["RECEIVE"]
PUBLISH = constants["PUBLISH"]
PUBLISH_FAILURE = constants["PUBLISH_FAILURE"]
PUBLISH_SUCCESS = constants["PUBLISH_SUCCESS"]
SUBSCRIPTION_FAILURE = constants["SUBSCRIPTION_FAILURE"]
SUBSCRIPTION_SUCCESS = constants["SUBSCRIPTION_SUCCESS"]
SUBSCRIPTION = constants["SUBSCRIPTION"]
UNSUBSCRIPTION = constants["UNSUBSCRIPTION"]
UNSUBSCRIPTION_FAILURE = constants["UNSUBSCRIPTION_FAILURE"]
UNSUBSCRIPTION_SUCCESS = constants["UNSUBSCRIPTION_SUCCESS"]
WEBCAM = constants["WEBCAM"]
CONFIGURE_TOPIC = constants["CONFIGURE_TOPIC"]
REGISTRATION_TOPIC = constants["REGISTRATION_TOPIC"]
PERMISSION_TOPIC = constants["PERMISSION_TOPIC"]
MODEL_FILE = constants["FACIALMODELFILE"]
OBJECT_URL = constants["OBJECT_URL"]

config_file_path = CWD_PATH + "/" + CONFIG_FILE
with open(config_file_path, 'r') as file:
    config = json.load(file)
UID = config["uid"]
ca_file = config["ca_file"]
cert = config["cert"]
key = config["key"]
endpoint = config["endpoint"]

cmdData = CommandLineUtils.parse_parameters(ca_file, cert, key, endpoint)
# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app,debug=True,cors_allowed_origins='*')
def on_connection_interrupted(connection, error):
    print("on_connection_interrupted")


# Callback when an interrupted connection is re-established.
def on_connection_resumed(connection, return_code, session_present, **kwargs):
    print("Connection resumed. return_code: {} session_present: {}".format(return_code, session_present))
    if return_code == mqtt.ConnectReturnCode.ACCEPTED and not session_present:
        print("Session did not persist. Resubscribing to existing topics...")


        # Cannot synchronously wait for resubscribe result because we're on the connection's event-loop thread,
        # evaluate result with a callback instead.
        # resubscribe_future.add_done_callback(on_resubscribe_complete)


def on_resubscribe_complete(resubscribe_future):
    resubscribe_results = resubscribe_future.result()
    print("Resubscribe results: {}".format(resubscribe_results))

    for topic, qos in resubscribe_results['topics']:
        if qos is None:
            #TODO: try the reconnection
            sys.exit("Server rejected resubscribe to topic: {}".format(topic))


if not cmdData.input_is_ci:
    print(f"Connecting to {cmdData.input_endpoint} with client ID '{cmdData.input_clientId}'...")
else:
    print("Connecting to endpoint with client ID")

    # Callback when the subscribed topic receives a message
    def on_message_received(topic, payload, dup, qos, retain, **kwargs):
        message_dict = json.loads(payload)
        print(message_dict)




# Callback when the connection successfully connects
def on_connection_success(connection, callback_data):
    assert isinstance(callback_data, mqtt.OnConnectionSuccessData)
    print("Connection Successful with return code: {} session present: {}".format(callback_data.return_code, callback_data.session_present))
        
# Callback when a connection attempt fails
def on_connection_failure(connection, callback_data):
    assert isinstance(callback_data, mqtt.OnConnectionFailureData)

# Callback when a connection has been disconnected or shutdown successfully
def on_connection_closed(connection, callback_data):
    print("Connection closed")

def on_subscribe_complete(subscribe_future):
    try:
        # Get the result of the subscription future
        subscribe_result = subscribe_future.result()
        for topic, qos in subscribe_result['topics']:
            print(topic, qos)
            # if qos is None:
            #     internal_queue.put((SUBSCRIPTION, topic, False))
            # else: 
            #     internal_queue.put((SUBSCRIPTION, topic, True))
    except Exception as e:
        print(f"Failed to subscribe to topic: {e}")


def on_publish_success(topic, message):
    # internal_queue.put((PUBLISH_SUCCESS, topic, message))
    print(PUBLISH_SUCCESS, topic, message)

def on_publish_failure(topic, message):
    # internal_queue.put((PUBLISH_FAILURE, topic, message))
    print(PUBLISH_FAILURE, topic, message)

def publish(self,topic, message):


    
    print(topic, message)
mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=cmdData.input_endpoint,
    port=cmdData.input_port,
    cert_filepath=cmdData.input_cert,
    pri_key_filepath=cmdData.input_key,
    ca_filepath=cmdData.input_ca,
    on_connection_interrupted=on_connection_interrupted,
    on_connection_resumed=on_connection_resumed,
    client_id=cmdData.input_clientId,
    clean_session=False,
    keep_alive_secs=30,
    http_proxy_options=None,
    on_connection_success=on_connection_success,
    on_connection_failure=on_connection_failure,
    on_connection_closed=on_connection_closed)


connect_future = mqtt_connection.connect()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO11 pose model
model_pose = YOLO("yolo11n-pose.pt")
model_detect = YOLO("yolo11n.pt")
DETECTED_LIST = ['laptop']
ACTIVE_KEYPOINT = [7,8,9,10,11,12]
FACE_THRESHOD = 90
CLASSIFIER_FOLDER = "classifier"
if not os.path.exists(CLASSIFIER_FOLDER):
    os.makedirs(CLASSIFIER_FOLDER)

houses = dict()

@app.route('/initial', methods=['POST'])
def setModel():
    try:
        houseId = request.form.get('houseId')
        url = request.form.get('modelUrl')
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            print("Ready to download....")
            modelName = f"{CLASSIFIER_FOLDER}/{houseId}.yml"
            with open(modelName, 'wb') as file:
                for chunk in response.iter_content(chunk_size=20* 1024 * 1024): 
                    file.write(chunk)
            print("download finish")
            return jsonify({'state': "success"}), 200
        else:
            return jsonify({'state': "failure"}), 400
    except Exception as e:
        print(f"error: {e}")
        return jsonify({'state': "failure"}), 400
    

# Route to handle multipart form data (file upload)
@app.route('/checkAlive', methods = ['GET'])
def checkAlive():
    return jsonify({'state': "sucess"}),200

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
            identity = float(identity) if identity is not None else identity
            detection = [float(x),float(y),float(w),float(h), identity]
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

def process_queue():
    while True:
        try:
            # Get an item from the queue
            data = task_queue.get()

            houseId = data['houseId']
            timeStamp = data['timeStamp']
            file_path = data['file_path']
            with open(file_path, 'rb') as f:
                image_data = f.read()  # Simulate processing the image
                print(f"Processing image for house {houseId}")
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Process the image and get the result
            result = process_image(img, houseId)
            
            # Check if houseId exists, create new house if needed
            if houseId not in houses:
                house = House(house_id=houseId)
                houses[houseId] = house
            else:
                house = houses[houseId]

            # Pass the result to the house for processing
            house.process_detection_data({
                'result': result,
                'timeStamp': timeStamp
            },mqtt_connection)

            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"remove{file_path}")

        except Exception as e:
            print(f"Error processing queue: {e}")
            time.sleep(1)

# Start the background queue processor in a thread when Flask starts
def start_background_queue_processor():
    thread = threading.Thread(target=process_queue)
    thread.daemon = True  # Ensures the thread will exit when the main program exits
    thread.start()

# API endpoint to submit a new image processing task
# @app.route('/submit', methods=['POST'])
# def submit_task():
#     houseId = request.form.get('houseId')
#     timeStamp = request.form.get('timeStamp')
#     frame = request.files.get('frame')

#     if houseId and frame  and timeStamp:
#         try:
#             # Ensure the 'photo' directory exists
#             photo_dir = 'photo'
#             if not os.path.exists(photo_dir):
#                 os.makedirs(photo_dir)

#             # Generate the filename and save the image to the 'photo' directory
#             formatted_timeStamp = str(timeStamp).replace('.', '_')
#             filename = f'frame_{houseId}_{formatted_timeStamp}.jpg'
#             file_path = os.path.join(photo_dir, filename)
#             frame.save(file_path)  # Save the binary image to the file system

#             # Add the task to the queue for background processing
#             task_queue.put({
#                 'houseId': houseId,
#                 'timeStamp': timeStamp,
#                 'file_path': file_path  # Pass the path to the saved image
#             })

#             # Return an immediate response to the client
#             return jsonify({"status": "Task received and is being processed", "file_saved": filename}), 202
#         except Exception as e:
#                 return jsonify({'error': str(e)}), 500

@socketio.on('submit_task', namespace="/submit")
def submit_task(data):
    houseId = data.get('houseId')
    timeStamp = data.get('timeStamp')
    frame = data.get('frame')  # Assuming frame is sent as binary data in base64
    print("receive photos")

    if houseId and frame and timeStamp:
        try:
            # Ensure the 'photo' directory exists
            photo_dir = 'photo'
            if not os.path.exists(photo_dir):
                os.makedirs(photo_dir)

            # Generate the filename and save the image to the 'photo' directory
            formatted_timeStamp = str(timeStamp).replace('.', '_')
            filename = f'frame_{houseId}_{formatted_timeStamp}.jpg'
            file_path = os.path.join(photo_dir, filename)

            # Decode the image if it's sent as base64, then save it
            with open(file_path, "wb") as f:
                f.write(frame)  # frame is expected to be binary data

            # Add the task to the queue for background processing
            task_queue.put({
                'houseId': houseId,
                'timeStamp': timeStamp,
                'file_path': file_path  # Pass the path to the saved image
            })

            # Send an acknowledgment back to the client
            emit("task_status", {"status": "Task received and is being processed", "file_saved": filename})
        except Exception as e:
            emit("task_status", {"error": str(e)})


if __name__ == '__main__':
   start_background_queue_processor()
   socketio.run(app, host='0.0.0.0', port=5000, debug=True)