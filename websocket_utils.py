import json
import cv2
import requests
import queue
import time
from datetime import datetime
import socketio.exceptions
import socketio
from sort import *
from collections import defaultdict
from collections import deque
from collections import Counter
from awscrt import mqtt, http
DETECTED_CLASSES=['people','laptop',"person"]
TRACKED_CLASS = ['people', 'object']
CWD_PATH = os.getcwd()
CONSTANT_FILE = "constants.json"
WD_PATH = os.getcwd()
CONFIG_FILE = "config.json"
config_file_path = CWD_PATH + "/" + CONFIG_FILE
with open(CWD_PATH + "/" + CONSTANT_FILE, 'r') as f:
    constants = json.load(f)
QOS = getattr(mqtt.QoS, constants["QOS"])
class Object_record:

    def __init__(self, tracked_id, class_name, mac_address = None):
        self.tracked_id = tracked_id  # Instance variable
        self.class_name = class_name
        self.state = False  # Instance variable
        self.user_id = None
        self.mac_address = mac_address
        self.not_seen_frame = 0
        self.touched_queue = deque(maxlen=10)

    # Method (function inside the class)
    def update_user(self, user_id):
        self.user_id = user_id
    def update_state(self,state):
        self.state = state
    def get_name(self):
        if self.mac_address:
            return self.mac_address
        else:
            return self.class_name
    def get_id(self):
        return self.tracked_id
    def get_state(self):
        return self.state
    def get_user(self):
        return self.user_id
    def get_class(self):
        return self.class_name
    def get_address(self):
        return self.mac_address
    def add_not_seen_frame(self):
        self.not_seen_frame += 1
    def init_not_seen_frame(self):
        self.not_seen_frame = 0
    def add_touched_list(self,user_id):
        self.touched_queue.append(user_id)
    def isUsing(self):
        touched_people = [p for p in self.touched_queue]
        
        person = touched_people[-1]
        if all(x is None for x in touched_people[-3:]):
            return None 
        if person:
            return person
        else:
            most = get_most(touched_people)
            return most

    def hasExpire(self):
        if self.not_seen_frame > 10000:
            return True
        else:
            return False


                



def get_most(lst):
    # Special case: if all elements in the list are None, return None
    if all(x is None for x in lst):
        return None

    # Filter out None values from the list
    filtered_lst = [x for x in lst if x is not None]

    # Count the occurrences of each element
    counter = Counter(filtered_lst)

    # Reverse the list to prioritize the last occurring element in case of a tie
    reverse_lst = filtered_lst[::-1]

    # Get the maximum count of occurrences
    max_count = max(counter.values())

    # Find the last element with the maximum count
    for key in reverse_lst:
        if counter[key] == max_count:
            return key



def get_using_situation(object_tracked_ids, people_tracked_ids, ppl_detections):
    using_dict = dict()
    for people_tracked_id, person in zip(people_tracked_ids, ppl_detections):
        _,_,_,_,_,index,*_ = person
        if index is None:
            continue
        object_tracked_id = object_tracked_ids[index]
        if not object_tracked_id or not people_tracked_id:
            continue
        using_dict[object_tracked_id] = people_tracked_id
    return using_dict


def process_results(result):
    objects = result[0]
    people = result[1]
    detections = dict()
    for obj in objects:
        detections.setdefault("object", []).append(obj)
    for person in people:
        detections.setdefault("people", []).append(person)
    return detections
    

def iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    # Compute intersection
    inter_width = max(0, x2_min - x1_max)
    inter_height = max(0, y2_min - y1_max)
    inter_area = inter_width * inter_height

    # Compute union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    # Return IoU
    if union_area == 0:
        return 0
    return inter_area / union_area

def match_tracked_detected(tracked_objects, detections_):
    matched_track_ids = []
    used_track_ids = set()  # Set to keep track of already matched track_ids

    # Iterate through each box in the original order
    boxes = [(x1,y1,x2,y2) for x1,y1,x2,y2,*_ in detections_]
    for original_box in boxes:
        best_iou = 0
        best_track_id = None
        
        # Find the tracked object with the highest IoU that hasn't been matched yet
        for tracked_obj in tracked_objects:
            tracked_box = tracked_obj[:4]  # Extract [x1, y1, x2, y2] from tracked objects
            track_id = tracked_obj[4]      # Extract Track ID
            
            if track_id in used_track_ids:
                continue  # Skip this track_id if it's already been used
            
            iou_value = iou(tracked_box, original_box)
            
            if iou_value > best_iou:
                best_iou = iou_value
                best_track_id = track_id  # Store the best matching track ID
        
        # Append the best track ID (or None if no match was found)
        matched_track_ids.append(best_track_id)
        
        # Mark the best matched track ID as used
        if best_track_id is not None:
            used_track_ids.add(best_track_id)

    return matched_track_ids


class House:
    def __init__(self,house_id):
        self.house_id = house_id
        self.tracked_objects = defaultdict(Object_record)  # Object_record is defined elsewhere
        self.tracked_people = defaultdict(lambda: deque(maxlen=10))  # Store people with a queue of identities
        self.timestamp =None  # Initialize with the current timestamp at creation
        self.trackers = self.initialize_trackers()  # Initialize trackers based on class names
        

    def initialize_trackers(self):
        """Initialize the trackers for each tracked class using the Sort tracker."""
        trackers = dict()
        for class_name in TRACKED_CLASS:
            trackers[class_name] = Sort()  # Sort is assumed to be imported from a tracking library
        return trackers

    def process_detection_data(self, data, mqtt_connection):
        """
        Process detection data, update tracking, handle object usage, and generate messages.
        :param data: The incoming detection data containing 'result' and 'timeStamp'.
        """
        
        
        result = data['result']
        timestamp = data['timeStamp']
        if(self.timestamp is not None and self.timestamp > timestamp):
            return
        self.timestamp = timestamp
        detections = process_results(result)
        trackings = self.update_trackers(detections)
        tracked_ids_dict = dict()

        # Match detections with trackings
        for class_name, detections_ in detections.items():
            if class_name in trackings.keys():
                trackings_ = trackings[class_name]
                tracked_ids = match_tracked_detected(trackings_, detections_)
                tracked_ids_dict[class_name] = tracked_ids
        
        # Update the identity of people
        if "people" in detections.keys():
            self.update_identity(tracked_ids_dict["people"], detections["people"])

        # Determine object usage if both people and objects are detected
        if "object" in detections.keys() and "people" in detections.keys():
            using_dict = get_using_situation(tracked_ids_dict['object'], tracked_ids_dict['people'], detections["people"])
        else:
            using_dict = dict()


        # Update tracked objects and generate messages for changes
        if "object" in detections.keys():
            self.update_object(using_dict, tracked_ids_dict['object'], detections["object"])

        # Generate and publish JSON messages for each object
        if "object" in detections.keys():
            for object_id in tracked_ids_dict['object']:
                self.generate_and_publish_message(object_id,mqtt_connection)

    def generate_and_publish_message(self, object_id,mqtt_connection):
        """Generate and publish JSON message for tracked object state changes."""
        object_record = self.tracked_objects[int(object_id)]
        current_state = object_record.get_state()
        using_people = object_record.isUsing()
        new_state = current_state

        if current_state:
            if using_people is None:
                new_state = False
                identity = object_record.get_user()
        else:
            if using_people is not None:
                new_state = True
                person_queue = self.tracked_people[int(using_people)]
                person_list = [p for p in person_queue]
                identity = get_most(person_list)
        # If the state has changed, update and send the message
        if new_state != current_state:
            object_record.update_state(new_state)
            if identity is None:
                identity = "unknown"
            if new_state:
                object_record.update_user(identity)
            else:
                object_record.update_user(None)
            
            
            # Create the message dictionary
            message_dict = {
                'identity': identity,
                'class': object_record.get_class(),
                'state': "on" if new_state else "off",
                'timeStamp': str(datetime.fromtimestamp(float(self.timestamp)).strftime("%Y-%m-%d %H:%M:%S"))
            }
            topic = f"houses/{self.house_id}"
            message_json = json.dumps(message_dict)
            mqtt_connection.publish(
            topic=topic,
            payload=message_json,
            qos=QOS
            )
            print(message_json)

    def update_trackers(self, detections):
        """Update the trackers with new detections."""
        trackings = dict()
        for class_name, detections_ in detections.items():
            tracker = self.get_tracker(class_name)
            if tracker:
                trackings_ = tracker.update(np.asarray([(x1, y1, x2, y2, score) for x1, y1, x2, y2, score, *_ in detections_])) if len(detections_) > 0 else []
                trackings[class_name] = trackings_
        return trackings

    def update_object(self, using_dict, object_tracked_ids, object_detections):
        """Update objects with usage and maintain tracking."""
        # Clean up old objects and update usage
        for object_id, object_record in self.tracked_objects.items():
            if object_id not in object_tracked_ids:
                object_record.add_not_seen_frame()
            else:
                object_record.init_not_seen_frame()

            if object_record.hasExpire():
                self.tracked_objects.pop(object_id)

        # Update the tracked objects based on detections and usage
        for idx, detection in enumerate(object_detections):
            track_id = object_tracked_ids[idx]
            if track_id in self.tracked_objects:
                object_record = self.tracked_objects[track_id]
            else:
                _, _, _, _, _, class_name = detection
                object_record = Object_record(track_id, class_name)
                self.tracked_objects[track_id] = object_record

            # If the object is being used by a person, update touched_list
            
            if track_id in using_dict.keys():
                user_id = using_dict[track_id]
                object_record.add_touched_list(user_id)
                print(using_dict, object_record.touched_queue)
            else:
                object_record.add_touched_list(None)
                print(using_dict, object_record.touched_queue)

    def get_tracker(self, class_name):
        """Get the tracker for a specific class name."""
        return self.trackers.get(class_name, None)

    def update_identity(self, tracked_ids, ppl_detections):
        """Update the identity of people being tracked."""
        for tracked_id, detection in zip(tracked_ids, ppl_detections):
            identity = detection[-1]
            if not tracked_id:
                continue
            if tracked_id in self.tracked_people.keys():
                queue = self.tracked_people[tracked_id]
            else:
                queue = deque(maxlen=10)
                self.tracked_people[tracked_id] = queue

            if identity is None and len(queue) > 0:
                continue
            else:
                queue.append(identity)

    def get_most(self, person_list):
        """Get the most frequent identity from a list of people."""
        return get_most(person_list) if person_list else None

