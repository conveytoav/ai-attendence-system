import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Dlib components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Database setup
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
current_date = datetime.datetime.now().strftime("%Y_%m_%d")
table_name = "attendance"
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)
conn.commit()
conn.close()

class SimplifiedFaceRecognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        
        # FPS tracking
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()
        self.frame_cnt = 0
        
        # Face database
        self.face_features_known_list = []
        self.face_name_known_list = []
        
        # Tracking variables
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        
        # Recognition variables
        self.current_frame_face_X_e_distance_list = []
        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []
        self.last_current_frame_centroid_e_distance = 0
        
        # Reclassification settings
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10
        
        # Recognition threshold - using only dlib features
        self.recognition_threshold = 0.4
        
        # Attendance tracking
        self.attendance_marked = set()
        
        logging.info("Simplified Face Recognizer initialized (Dlib only)")
    
    def extract_dlib_features(self, img, shape):
        """Extract 128D features using Dlib"""
        try:
            face_descriptor = face_reco_model.compute_face_descriptor(img, shape)
            return np.array(face_descriptor)
        except Exception as e:
            logging.error(f"Error extracting Dlib features: {e}")
            return np.zeros(128)
    
    def extract_face_features(self, img):
        """Extract face features using only Dlib"""
        # Detect faces
        faces = detector(img, 1)
        if len(faces) == 0:
            return np.zeros(128)
        
        # Get the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get facial landmarks
        shape = predictor(img, face)
        
        # Extract features
        features = self.extract_dlib_features(img, shape)
        return features
    
    def get_face_database(self):
        """Load face database from CSV file"""
        # Try different CSV files
        csv_files = ["data/features_all.csv", "data/features_all_enhanced.csv"]
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    csv_rd = pd.read_csv(csv_file)
                    
                    for i in range(csv_rd.shape[0]):
                        # Get person name
                        person_name = csv_rd.iloc[i, 0]  # First column is person name
                        self.face_name_known_list.append(person_name)
                        
                        # Get features (first 128 features for dlib compatibility)
                        features = csv_rd.iloc[i, 1:129].values  # Only take first 128 features
                        
                        # Handle missing values
                        features = np.where(pd.isna(features), 0, features)
                        
                        self.face_features_known_list.append(features.astype(float))
                    
                    logging.info(f"Loaded {len(self.face_features_known_list)} faces from {csv_file}")
                    logging.info(f"Feature dimensions: {len(self.face_features_known_list[0])}")
                    return True
                    
                except Exception as e:
                    logging.error(f"Error loading {csv_file}: {e}")
                    continue
        
        # If no CSV found, try to create a simple database
        logging.warning("No face database found. Creating empty database.")
        return True
    
    def update_fps(self):
        """Update FPS calculation"""
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
    
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        """Compute euclidean distance between two feature vectors"""
        feature_1 = np.array(feature_1, dtype=np.float64)
        feature_2 = np.array(feature_2, dtype=np.float64)
        
        # Handle different feature lengths
        min_len = min(len(feature_1), len(feature_2))
        feature_1 = feature_1[:min_len]
        feature_2 = feature_2[:min_len]
        
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist
    
    def centroid_tracker(self):
        """Track faces using centroid positions"""
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], 
                    self.last_frame_face_centroid_list[j]
                )
                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance
                )
            
            if e_distance_current_frame_person_x_list:
                last_frame_num = e_distance_current_frame_person_x_list.index(
                    min(e_distance_current_frame_person_x_list)
                )
                self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]
    
    def draw_note(self, img_rd):
        """Draw information on the frame"""
        cv2.putText(img_rd, "Simplified Face Recognizer (Dlib)", (20, 40), 
                    self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"Frame: {self.frame_cnt}", (20, 100), 
                    self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"FPS: {self.fps.__round__(2)}", (20, 130), 
                    self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"Faces: {self.current_frame_face_cnt}", (20, 160), 
                    self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, f"Threshold: {self.recognition_threshold}", (20, 190), 
                    self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), 
                    self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw face labels
        for i in range(len(self.current_frame_face_name_list)):
            cv2.putText(img_rd, f"Face_{i + 1}", 
                        tuple([int(self.current_frame_face_centroid_list[i][0]), 
                              int(self.current_frame_face_centroid_list[i][1])]),
                        self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)
    
    def attendance(self, name):
        """Mark attendance in database"""
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Check if already marked today
        attendance_key = f"{name}_{current_date}"
        if attendance_key in self.attendance_marked:
            return
        
        try:
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            # Check if already exists in database
            cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", 
                          (name, current_date))
            existing_entry = cursor.fetchone()
            
            if existing_entry:
                logging.info(f"{name} already marked present for {current_date}")
            else:
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", 
                              (name, current_time, current_date))
                conn.commit()
                logging.info(f"{name} marked present for {current_date} at {current_time}")
            
            # Add to local tracking
            self.attendance_marked.add(attendance_key)
            conn.close()
            
        except Exception as e:
            logging.error(f"Error marking attendance for {name}: {e}")
    
    def process(self, stream):
        """Main processing loop"""
        if not self.get_face_database():
            return
        
        logging.info("Starting face recognition...")
        
        while stream.isOpened():
            self.frame_cnt += 1
            logging.debug(f"Frame {self.frame_cnt} starts")
            
            flag, img_rd = stream.read()
            if not flag:
                break
            
            kk = cv2.waitKey(1)
            
            # Detect faces
            faces = detector(img_rd, 0)
            
            # Update face counts
            self.last_frame_face_cnt = self.current_frame_face_cnt
            self.current_frame_face_cnt = len(faces)
            
            # Update face name list
            self.last_frame_face_name_list = self.current_frame_face_name_list[:]
            
            # Update centroid list
            self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
            self.current_frame_face_centroid_list = []
            
            # Process based on face count changes
            if (self.current_frame_face_cnt == self.last_frame_face_cnt and 
                self.reclassify_interval_cnt != self.reclassify_interval):
                # No change in face count
                logging.debug("No face count changes")
                
                self.current_frame_face_position_list = []
                
                if "unknown" in self.current_frame_face_name_list:
                    self.reclassify_interval_cnt += 1
                
                if self.current_frame_face_cnt != 0:
                    for k, d in enumerate(faces):
                        self.current_frame_face_position_list.append(tuple([
                            faces[k].left(), 
                            int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)
                        ]))
                        
                        self.current_frame_face_centroid_list.append([
                            int(faces[k].left() + faces[k].right()) / 2,
                            int(faces[k].top() + faces[k].bottom()) / 2
                        ])
                        
                        # Draw rectangle
                        cv2.rectangle(img_rd, 
                                    tuple([d.left(), d.top()]),
                                    tuple([d.right(), d.bottom()]),
                                    (255, 255, 255), 2)
                
                # Use centroid tracker for multiple faces
                if self.current_frame_face_cnt > 1:
                    self.centroid_tracker()
                
                # Draw names
                for i in range(self.current_frame_face_cnt):
                    cv2.putText(img_rd, self.current_frame_face_name_list[i],
                               self.current_frame_face_position_list[i], 
                               self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            
            else:
                # Face count changed - perform recognition
                logging.debug("Face count changed - performing recognition")
                
                self.current_frame_face_position_list = []
                self.current_frame_face_X_e_distance_list = []
                self.current_frame_face_feature_list = []
                self.reclassify_interval_cnt = 0
                
                if self.current_frame_face_cnt == 0:
                    logging.debug("No faces in frame")
                    self.current_frame_face_name_list = []
                else:
                    logging.debug(f"Processing {self.current_frame_face_cnt} faces")
                    self.current_frame_face_name_list = []
                    
                    # Extract features for each face
                    for i in range(len(faces)):
                        features = self.extract_face_features(img_rd)
                        self.current_frame_face_feature_list.append(features)
                        self.current_frame_face_name_list.append("unknown")
                    
                    # Perform recognition
                    for k in range(len(faces)):
                        logging.debug(f"Recognizing face {k + 1}")
                        
                        # Update centroid and position
                        self.current_frame_face_centroid_list.append([
                            int(faces[k].left() + faces[k].right()) / 2,
                            int(faces[k].top() + faces[k].bottom()) / 2
                        ])
                        
                        self.current_frame_face_position_list.append(tuple([
                            faces[k].left(),
                            int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)
                        ]))
                        
                        # Compare with known faces
                        if len(self.face_features_known_list) > 0:
                            self.current_frame_face_X_e_distance_list = []
                            
                            for i in range(len(self.face_features_known_list)):
                                try:
                                    e_distance = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i]
                                    )
                                    self.current_frame_face_X_e_distance_list.append(e_distance)
                                    logging.debug(f"Distance to {self.face_name_known_list[i]}: {e_distance}")
                                except Exception as e:
                                    logging.error(f"Error computing distance: {e}")
                                    self.current_frame_face_X_e_distance_list.append(999999999)
                            
                            # Find best match
                            if self.current_frame_face_X_e_distance_list:
                                min_distance = min(self.current_frame_face_X_e_distance_list)
                                similar_person_num = self.current_frame_face_X_e_distance_list.index(min_distance)
                                
                                if min_distance < self.recognition_threshold:
                                    recognized_name = self.face_name_known_list[similar_person_num]
                                    self.current_frame_face_name_list[k] = recognized_name
                                    logging.info(f"Recognized: {recognized_name} (distance: {min_distance:.3f})")
                                    
                                    # Mark attendance
                                    self.attendance(recognized_name)
                                else:
                                    logging.debug(f"Unknown person (min distance: {min_distance:.3f})")
                        
                        # Draw rectangle
                        cv2.rectangle(img_rd,
                                    tuple([faces[k].left(), faces[k].top()]),
                                    tuple([faces[k].right(), faces[k].bottom()]),
                                    (255, 255, 255), 2)
                        
                        # Draw name
                        cv2.putText(img_rd, self.current_frame_face_name_list[k],
                                   self.current_frame_face_position_list[k],
                                   self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Draw information
            self.draw_note(img_rd)
            
            # Exit on 'q'
            if kk == ord('q'):
                break
            
            # Update FPS and display
            self.update_fps()
            cv2.namedWindow("Simplified Face Recognition", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Simplified Face Recognition", img_rd)
            
            logging.debug("Frame processing complete\n")
    
    def run(self):
        """Run the face recognition system"""
        cap = cv2.VideoCapture(0)  # Use camera
        
        if not cap.isOpened():
            logging.error("Could not open camera")
            return
        
        try:
            self.process(cap)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Face recognition system stopped")

def main():
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    recognizer = SimplifiedFaceRecognizer()
    recognizer.run()

if __name__ == '__main__':
    main()