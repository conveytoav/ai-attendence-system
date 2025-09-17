import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont, filedialog, messagebox
from PIL import Image, ImageTk
import face_recognition
from sklearn.preprocessing import StandardScaler
import pickle

# Use frontal face detector of Dlib with shape predictor for better accuracy
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib
face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Download from dlib


class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        
        # Enhanced face detection parameters
        self.face_quality_threshold = 0.6  # Minimum face quality score
        self.min_face_size = 50  # Minimum face size in pixels
        self.max_face_size = 300  # Maximum face size in pixels
        
        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Enhanced Face Register")
        self.win.geometry("1200x700")
        self.win.configure(bg='#f0f0f0')

        # GUI left part
        self.frame_left_camera = tk.Frame(self.win, bg='#f0f0f0')
        self.label = tk.Label(self.win, bg='#f0f0f0')
        self.label.pack(side=tk.LEFT, padx=20, pady=20)
        self.frame_left_camera.pack()

        # GUI right part
        self.frame_right_info = tk.Frame(self.win, bg='#f0f0f0')
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt), bg='#f0f0f0')
        self.label_fps_info = tk.Label(self.frame_right_info, text="", bg='#f0f0f0')
        self.input_name = tk.Entry(self.frame_right_info, font=('Arial', 12), width=20)
        self.input_name_char = ""
        self.label_warning = tk.Label(self.frame_right_info, bg='#f0f0f0')
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ", bg='#f0f0f0')
        self.log_all = tk.Label(self.frame_right_info, bg='#f0f0f0', wraplength=300, justify=tk.LEFT)
        
        # Quality indicators
        self.label_face_quality = tk.Label(self.frame_right_info, text="Face Quality: ", bg='#f0f0f0')
        self.label_brightness = tk.Label(self.frame_right_info, text="Brightness: ", bg='#f0f0f0')
        self.label_sharpness = tk.Label(self.frame_right_info, text="Sharpness: ", bg='#f0f0f0')

        self.font_title = tkFont.Font(family='Arial', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Arial', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Arial', size=12, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Current frame and face ROI position
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False
        self.uploaded_image = None
        self.using_uploaded_image = False

        # Enhanced quality metrics
        self.face_quality_score = 0.0
        self.brightness_score = 0.0
        self.sharpness_score = 0.0

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # Camera setup with better parameters
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    def calculate_face_quality(self, face_image):
        """Calculate face quality based on multiple metrics"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            
            # Brightness check
            brightness = np.mean(gray)
            brightness_score = min(1.0, brightness / 128.0) if brightness < 128 else min(1.0, (255 - brightness) / 128.0)
            
            # Sharpness check using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)
            
            # Face size check
            face_size = min(face_image.shape[:2])
            size_score = 1.0 if self.min_face_size <= face_size <= self.max_face_size else 0.5
            
            # Overall quality score
            quality_score = (brightness_score + sharpness_score + size_score) / 3.0
            
            return quality_score, brightness_score, sharpness_score
            
        except Exception as e:
            logging.error(f"Error calculating face quality: {e}")
            return 0.0, 0.0, 0.0

    def detect_faces_enhanced(self, image):
        """Enhanced face detection with quality assessment"""
        try:
            # Use both dlib and face_recognition for better accuracy
            faces_dlib = detector(image, 1)  # Higher upsampling for better detection
            faces_fr = face_recognition.face_locations(image, model="hog")
            
            # Combine results and filter by quality
            quality_faces = []
            
            for face in faces_dlib:
                # Extract face region
                face_img = image[face.top():face.bottom(), face.left():face.right()]
                if face_img.size > 0:
                    quality, brightness, sharpness = self.calculate_face_quality(face_img)
                    if quality >= self.face_quality_threshold:
                        quality_faces.append((face, quality, brightness, sharpness))
            
            return quality_faces
            
        except Exception as e:
            logging.error(f"Error in enhanced face detection: {e}")
            return []

    def upload_image(self):
        """Upload image for face registration"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Load and process image
                image = cv2.imread(file_path)
                if image is not None:
                    # Resize if too large
                    height, width = image.shape[:2]
                    if width > 1280 or height > 720:
                        scale = min(1280/width, 720/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        image = cv2.resize(image, (new_width, new_height))
                    
                    self.uploaded_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.using_uploaded_image = True
                    self.log_all["text"] = f"Image uploaded successfully!\nFile: {os.path.basename(file_path)}"
                    
                    # Process the uploaded image immediately
                    self.process_uploaded_image()
                else:
                    messagebox.showerror("Error", "Could not load the selected image.")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error uploading image: {e}")
            logging.error(f"Error uploading image: {e}")

    def process_uploaded_image(self):
        """Process uploaded image for face detection"""
        if self.uploaded_image is not None:
            self.current_frame = self.uploaded_image.copy()
            
            # Detect faces with enhanced method
            quality_faces = self.detect_faces_enhanced(self.current_frame)
            
            if quality_faces:
                # Use the best quality face
                best_face = max(quality_faces, key=lambda x: x[1])
                face, quality, brightness, sharpness = best_face
                
                # Update quality metrics
                self.face_quality_score = quality
                self.brightness_score = brightness
                self.sharpness_score = sharpness
                
                # Update face position
                self.face_ROI_width_start = face.left()
                self.face_ROI_height_start = face.top()
                self.face_ROI_height = face.bottom() - face.top()
                self.face_ROI_width = face.right() - face.left()
                self.hh = int(self.face_ROI_height / 2)
                self.ww = int(self.face_ROI_width / 2)
                
                # Draw rectangle around face
                color = (0, 255, 0) if quality >= self.face_quality_threshold else (255, 165, 0)
                cv2.rectangle(self.current_frame,
                            (face.left() - self.ww, face.top() - self.hh),
                            (face.right() + self.ww, face.bottom() + self.hh),
                            color, 2)
                
                # Add quality text
                cv2.putText(self.current_frame, f"Quality: {quality:.2f}", 
                           (face.left(), face.top() - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                self.current_frame_faces_cnt = 1
                self.out_of_range_flag = False
                
            else:
                self.current_frame_faces_cnt = 0
                self.log_all["text"] = "No suitable faces found in uploaded image."
            
            # Update display
            self.update_quality_display()
            self.update_display()

    def update_quality_display(self):
        """Update quality indicators in GUI"""
        self.label_face_quality["text"] = f"Face Quality: {self.face_quality_score:.2f}"
        self.label_brightness["text"] = f"Brightness: {self.brightness_score:.2f}"
        self.label_sharpness["text"] = f"Sharpness: {self.sharpness_score:.2f}"
        
        # Color coding for quality
        if self.face_quality_score >= 0.8:
            self.label_face_quality['fg'] = 'green'
        elif self.face_quality_score >= 0.6:
            self.label_face_quality['fg'] = 'orange'
        else:
            self.label_face_quality['fg'] = 'red'

    def update_display(self):
        """Update the image display"""
        if self.current_frame is not None:
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

    def switch_to_camera(self):
        """Switch back to camera mode"""
        self.using_uploaded_image = False
        self.uploaded_image = None
        self.log_all["text"] = "Switched to camera mode"

    def GUI_clear_data(self):
        """Delete old face folders"""
        try:
            if os.path.exists(self.path_photos_from_camera):
                folders_rd = os.listdir(self.path_photos_from_camera)
                for folder in folders_rd:
                    folder_path = os.path.join(self.path_photos_from_camera, folder)
                    if os.path.isdir(folder_path):
                        shutil.rmtree(folder_path)
                        
            if os.path.isfile("data/features_all.csv"):
                os.remove("data/features_all.csv")
                
            if os.path.isfile("data/face_encodings.pkl"):
                os.remove("data/face_encodings.pkl")
                
            self.label_cnt_face_in_database['text'] = "0"
            self.existing_faces_cnt = 0
            self.log_all["text"] = "All face data cleared successfully!"
            
        except Exception as e:
            self.log_all["text"] = f"Error clearing data: {e}"
            logging.error(f"Error clearing data: {e}")

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get().strip()
        if self.input_name_char:
            self.create_face_folder()
            self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)
        else:
            self.log_all["text"] = "Please enter a valid name!"

    def GUI_info(self):
        """Create enhanced GUI layout"""
        # Title
        tk.Label(self.frame_right_info,
                text="Enhanced Face Register",
                font=self.font_title,
                bg='#f0f0f0').grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=20)

        # System info
        tk.Label(self.frame_right_info, text="FPS: ", bg='#f0f0f0').grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Faces in database: ", bg='#f0f0f0').grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Faces in current frame: ", bg='#f0f0f0').grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)

        # Quality indicators
        self.label_face_quality.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        self.label_brightness.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        self.label_sharpness.grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                font=self.font_step_title,
                text="Step 1: Clear face photos",
                bg='#f0f0f0').grid(row=8, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(20,5))
        tk.Button(self.frame_right_info,
                 text='Clear Database',
                 command=self.GUI_clear_data,
                 bg='#ff6b6b',
                 fg='white',
                 font=('Arial', 10, 'bold')).grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 2: Input name
        tk.Label(self.frame_right_info,
                font=self.font_step_title,
                text="Step 2: Input name",
                bg='#f0f0f0').grid(row=10, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(20,5))

        tk.Label(self.frame_right_info, text="Name: ", bg='#f0f0f0').grid(row=11, column=0, sticky=tk.W, padx=5, pady=2)
        self.input_name.grid(row=11, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Button(self.frame_right_info,
                 text='Create Folder',
                 command=self.GUI_get_input_name,
                 bg='#4ecdc4',
                 fg='white',
                 font=('Arial', 10, 'bold')).grid(row=11, column=2, padx=5, pady=2)

        # Step 3: Choose input method
        tk.Label(self.frame_right_info,
                font=self.font_step_title,
                text="Step 3: Choose input method",
                bg='#f0f0f0').grid(row=12, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(20,5))

        tk.Button(self.frame_right_info,
                 text='Use Camera',
                 command=self.switch_to_camera,
                 bg='#45b7d1',
                 fg='white',
                 font=('Arial', 10, 'bold')).grid(row=13, column=0, sticky=tk.W, padx=5, pady=2)

        tk.Button(self.frame_right_info,
                 text='Upload Image',
                 command=self.upload_image,
                 bg='#96ceb4',
                 fg='white',
                 font=('Arial', 10, 'bold')).grid(row=13, column=1, sticky=tk.W, padx=5, pady=2)

        # Step 4: Save face
        tk.Label(self.frame_right_info,
                font=self.font_step_title,
                text="Step 4: Save face image",
                bg='#f0f0f0').grid(row=14, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(20,5))

        tk.Button(self.frame_right_info,
                 text='Save Current Face',
                 command=self.save_current_face,
                 bg='#feca57',
                 fg='white',
                 font=('Arial', 10, 'bold')).grid(row=15, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Auto-save option
        tk.Button(self.frame_right_info,
                 text='Auto-Save (10 photos)',
                 command=self.auto_save_faces,
                 bg='#ff9ff3',
                 fg='white',
                 font=('Arial', 10, 'bold')).grid(row=16, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Log display
        self.log_all.grid(row=17, column=0, columnspan=3, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

    def auto_save_faces(self):
        """Automatically save multiple face images"""
        if not self.face_folder_created_flag:
            self.log_all["text"] = "Please create a folder first (Step 2)!"
            return
            
        self.auto_save_count = 0
        self.auto_save_target = 10
        self.auto_save_active = True
        self.log_all["text"] = f"Auto-save started. Target: {self.auto_save_target} photos"

    def pre_work_mkdir(self):
        """Create necessary directories"""
        os.makedirs(self.path_photos_from_camera, exist_ok=True)
        os.makedirs("data", exist_ok=True)

    def check_existing_faces_cnt(self):
        """Check existing faces count"""
        try:
            if os.path.exists(self.path_photos_from_camera) and os.listdir(self.path_photos_from_camera):
                person_list = os.listdir(self.path_photos_from_camera)
                person_num_list = []
                for person in person_list:
                    if person.startswith("person_"):
                        try:
                            person_order = person.split('_')[1]
                            person_num_list.append(int(person_order))
                        except (IndexError, ValueError):
                            continue
                if person_num_list:
                    self.existing_faces_cnt = max(person_num_list)
                else:
                    self.existing_faces_cnt = 0
            else:
                self.existing_faces_cnt = 0
        except Exception as e:
            logging.error(f"Error checking existing faces: {e}")
            self.existing_faces_cnt = 0

    def update_fps(self):
        """Update FPS calculation"""
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time if self.frame_time > 0 else 0
        self.frame_start_time = now
        self.label_fps_info["text"] = str(round(self.fps, 2))

    def create_face_folder(self):
        """Create folder for saving faces"""
        try:
            self.existing_faces_cnt += 1
            if self.input_name_char:
                self.current_face_dir = os.path.join(
                    self.path_photos_from_camera,
                    f"person_{self.existing_faces_cnt}_{self.input_name_char}"
                )
            else:
                self.current_face_dir = os.path.join(
                    self.path_photos_from_camera,
                    f"person_{self.existing_faces_cnt}"
                )
            
            os.makedirs(self.current_face_dir, exist_ok=True)
            self.log_all["text"] = f"Folder created: {self.current_face_dir}"
            logging.info(f"Created folder: {self.current_face_dir}")
            
            self.ss_cnt = 0
            self.face_folder_created_flag = True
            
        except Exception as e:
            self.log_all["text"] = f"Error creating folder: {e}"
            logging.error(f"Error creating folder: {e}")

    def save_current_face(self):
        """Save current face with enhanced quality"""
        if not self.face_folder_created_flag:
            self.log_all["text"] = "Please create a folder first (Step 2)!"
            return
            
        if self.current_frame_faces_cnt != 1:
            self.log_all["text"] = "Please ensure exactly one face is visible!"
            return
            
        if self.out_of_range_flag:
            self.log_all["text"] = "Face is out of range!"
            return
            
        if self.face_quality_score < self.face_quality_threshold:
            self.log_all["text"] = f"Face quality too low: {self.face_quality_score:.2f}"
            return
            
        try:
            self.ss_cnt += 1
            
            # Extract face with larger margin for better quality
            margin = max(self.ww, self.hh)
            y_start = max(0, self.face_ROI_height_start - margin)
            y_end = min(self.current_frame.shape[0], self.face_ROI_height_start + self.face_ROI_height + margin)
            x_start = max(0, self.face_ROI_width_start - margin)
            x_end = min(self.current_frame.shape[1], self.face_ROI_width_start + self.face_ROI_width + margin)
            
            self.face_ROI_image = self.current_frame[y_start:y_end, x_start:x_end]
            
            # Resize to standard size for consistency
            self.face_ROI_image = cv2.resize(self.face_ROI_image, (224, 224))
            
            # Save image
            filename = f"img_face_{self.ss_cnt:03d}.jpg"
            filepath = os.path.join(self.current_face_dir, filename)
            
            # Convert RGB to BGR for saving
            face_bgr = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, face_bgr)
            
            self.log_all["text"] = f"Saved: {filename}\nQuality: {self.face_quality_score:.2f}"
            logging.info(f"Saved face image: {filepath}")
            
        except Exception as e:
            self.log_all["text"] = f"Error saving face: {e}"
            logging.error(f"Error saving face: {e}")

    def get_frame(self):
        """Get frame from camera"""
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (480, 360))
                    return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Error getting frame: {e}")
        return False, None

    def process(self):
        """Main processing loop"""
        if not self.using_uploaded_image:
            ret, self.current_frame = self.get_frame()
            if ret:
                self.update_fps()
                
                # Enhanced face detection
                quality_faces = self.detect_faces_enhanced(self.current_frame)
                
                if quality_faces:
                    # Use the best quality face
                    best_face = max(quality_faces, key=lambda x: x[1])
                    face, quality, brightness, sharpness = best_face
                    
                    # Update metrics
                    self.face_quality_score = quality
                    self.brightness_score = brightness
                    self.sharpness_score = sharpness
                    
                    # Update face position
                    self.face_ROI_width_start = face.left()
                    self.face_ROI_height_start = face.top()
                    self.face_ROI_height = face.bottom() - face.top()
                    self.face_ROI_width = face.right() - face.left()
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)
                    
                    # Check if face is in range
                    if (face.right() + self.ww > 640 or face.bottom() + self.hh > 480 or 
                        face.left() - self.ww < 0 or face.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color = (0, 255, 0) if quality >= self.face_quality_threshold else (255, 165, 0)
                    
                    # Draw rectangle and quality info
                    cv2.rectangle(self.current_frame,
                                (face.left() - self.ww, face.top() - self.hh),
                                (face.right() + self.ww, face.bottom() + self.hh),
                                color, 2)
                    
                    cv2.putText(self.current_frame, f"Q: {quality:.2f}", 
                               (face.left(), face.top() - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    self.current_frame_faces_cnt = len(quality_faces)
                    
                    # Auto-save functionality
                    if hasattr(self, 'auto_save_active') and self.auto_save_active:
                        if (quality >= self.face_quality_threshold and 
                            not self.out_of_range_flag and 
                            self.face_folder_created_flag):
                            self.save_current_face()
                            self.auto_save_count += 1
                            if self.auto_save_count >= self.auto_save_target:
                                self.auto_save_active = False
                                self.log_all["text"] = f"Auto-save completed! Saved {self.auto_save_count} photos."
                else:
                    self.current_frame_faces_cnt = 0
                    self.face_quality_score = 0.0
                    self.brightness_score = 0.0
                    self.sharpness_score = 0.0
                
                # Update quality display
                self.update_quality_display()
                self.label_face_cnt["text"] = str(self.current_frame_faces_cnt)
                
                # Update display
                self.update_display()
        
        # Schedule next frame
        self.win.after(20, self.process)

    def run(self):
        """Run the application"""
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    
    # Check for required model files
    required_files = [
        "shape_predictor_68_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required model files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download these files from:")
        print("  - shape_predictor_68_face_landmarks.dat: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("  - dlib_face_recognition_resnet_model_v1.dat: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
        print("\nExtract the .bz2 files and place them in the same directory as this script.")
        return
    
    try:
        Face_Register_con = Face_Register()
        Face_Register_con.run()
    except Exception as e:
        logging.error(f"Error running application: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()