# Enhanced Face Feature Extraction with CNN and Improved Accuracy
import os
import dlib
import csv
import numpy as np
import logging
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

class EnhancedFaceFeatureExtractor:
    def __init__(self):
        # Initialize Dlib components
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        
        # Initialize CNN models
        self.vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.pca = None  # Will be initialized dynamically based on data
        
        # Paths
        self.path_images_from_camera = "data/data_faces_from_camera/"
        self.features_file = "data/features_all_enhanced.csv"
        self.scaler_file = "data/scaler.pkl"
        self.pca_file = "data/pca.pkl"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def detect_and_align_face(self, img):
        """Detect and align face for better feature extraction"""
        faces = self.detector(img, 1)
        if len(faces) == 0:
            return None, None
        
        # Get the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get facial landmarks
        shape = self.predictor(img, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        # Align face based on eye positions
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        
        # Calculate angle and align
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Get face center
        face_center = ((face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2)
        
        # Rotate image
        M = cv2.getRotationMatrix2D(face_center, angle, 1.0)
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Extract aligned face
        margin = 20
        x1 = max(0, face.left() - margin)
        y1 = max(0, face.top() - margin)
        x2 = min(img.shape[1], face.right() + margin)
        y2 = min(img.shape[0], face.bottom() + margin)
        
        face_crop = aligned[y1:y2, x1:x2]
        
        return face_crop, shape
    
    def extract_dlib_features(self, img, shape):
        """Extract 128D features using Dlib"""
        try:
            face_descriptor = self.face_reco_model.compute_face_descriptor(img, shape)
            return np.array(face_descriptor)
        except Exception as e:
            self.logger.error(f"Error extracting Dlib features: {e}")
            return np.zeros(128)
    
    def extract_cnn_features(self, face_crop):
        """Extract CNN features using VGG16 and ResNet50"""
        if face_crop is None or face_crop.size == 0:
            return np.zeros(2048 + 2048)  # VGG16 + ResNet50 features
        
        try:
            # Resize face for CNN input
            face_resized = cv2.resize(face_crop, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = img_to_array(face_rgb)
            face_batch = np.expand_dims(face_array, axis=0)
            
            # VGG16 features
            vgg_input = vgg_preprocess(face_batch.copy())
            vgg_features = self.vgg_model.predict(vgg_input, verbose=0).flatten()
            
            # ResNet50 features
            resnet_input = resnet_preprocess(face_batch.copy())
            resnet_features = self.resnet_model.predict(resnet_input, verbose=0).flatten()
            
            # Combine features
            combined_features = np.concatenate([vgg_features, resnet_features])
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error extracting CNN features: {e}")
            return np.zeros(2048 + 2048)
    
    def extract_enhanced_features(self, img_path):
        """Extract enhanced features combining Dlib and CNN"""
        img = cv2.imread(img_path)
        if img is None:
            self.logger.error(f"Could not read image: {img_path}")
            return np.zeros(128 + 4096)  # Dlib + CNN features
        
        # Detect and align face
        face_crop, shape = self.detect_and_align_face(img)
        
        if face_crop is None:
            self.logger.warning(f"No face detected in: {img_path}")
            return np.zeros(128 + 4096)
        
        # Extract Dlib features
        dlib_features = self.extract_dlib_features(img, shape)
        
        # Extract CNN features
        cnn_features = self.extract_cnn_features(face_crop)
        
        # Combine features
        combined_features = np.concatenate([dlib_features, cnn_features])
        return combined_features
    
    def process_person_images(self, person_folder):
        """Process all images for a person and return average features"""
        person_path = os.path.join(self.path_images_from_camera, person_folder)
        
        if not os.path.exists(person_path):
            self.logger.warning(f"Person folder not found: {person_path}")
            return np.zeros(128 + 4096)
        
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            self.logger.warning(f"No images found in: {person_path}")
            return np.zeros(128 + 4096)
        
        features_list = []
        valid_images = 0
        
        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            self.logger.info(f"Processing: {img_path}")
            
            features = self.extract_enhanced_features(img_path)
            
            # Check if valid features were extracted
            if not np.allclose(features, 0):
                features_list.append(features)
                valid_images += 1
            else:
                self.logger.warning(f"No valid features extracted from: {img_path}")
        
        if not features_list:
            self.logger.error(f"No valid features extracted for person: {person_folder}")
            return np.zeros(128 + 4096)
        
        self.logger.info(f"Successfully processed {valid_images}/{len(image_files)} images for {person_folder}")
        
        # Calculate mean features with outlier removal
        features_array = np.array(features_list)
        
        # Remove outliers using z-score
        z_scores = np.abs((features_array - np.mean(features_array, axis=0)) / np.std(features_array, axis=0))
        mask = np.all(z_scores < 3, axis=1)  # Keep features within 3 standard deviations
        
        if np.any(mask):
            features_array = features_array[mask]
            self.logger.info(f"Removed {np.sum(~mask)} outlier images for {person_folder}")
        
        return np.mean(features_array, axis=0)
    
    def fit_preprocessing_components(self, all_features):
        """Fit scaler and PCA on all features"""
        # Separate Dlib and CNN features
        dlib_features = all_features[:, :128]
        cnn_features = all_features[:, 128:]
        
        n_samples = cnn_features.shape[0]
        n_features = cnn_features.shape[1]
        
        # Fit scaler on CNN features
        self.scaler.fit(cnn_features)
        
        # Determine optimal number of PCA components
        # Can't exceed min(n_samples, n_features)
        max_components = min(n_samples, n_features)
        
        if max_components > 1:
            # Use minimum of desired components (256) and maximum possible
            n_components = min(256, max_components)
            self.pca = PCA(n_components=n_components)
            
            # Transform and fit PCA
            cnn_scaled = self.scaler.transform(cnn_features)
            self.pca.fit(cnn_scaled)
            
            self.logger.info(f"PCA initialized with {n_components} components (max possible: {max_components})")
        else:
            # If only one sample, don't use PCA
            self.pca = None
            self.logger.info("PCA skipped due to insufficient samples (n_samples < 2)")
        
        # Save preprocessing components
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.pca_file, 'wb') as f:
            pickle.dump(self.pca, f)
        
        self.logger.info("Preprocessing components fitted and saved")
    
    def process_all_features(self, all_features):
        """Process all features with scaling and PCA"""
        # Separate Dlib and CNN features
        dlib_features = all_features[:, :128]
        cnn_features = all_features[:, 128:]
        
        # Scale CNN features
        cnn_scaled = self.scaler.transform(cnn_features)
        
        # Apply PCA to CNN features if available
        if self.pca is not None:
            cnn_processed = self.pca.transform(cnn_scaled)
        else:
            # If no PCA, just use scaled features
            cnn_processed = cnn_scaled
        
        # Combine processed features
        processed_features = np.hstack([dlib_features, cnn_processed])
        
        return processed_features
    
    def save_features_to_csv(self, person_names, features_array):
        """Save features to CSV file"""
        with open(self.features_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Determine feature dimensions
            dlib_features = 128
            cnn_features = features_array.shape[1] - dlib_features
            
            # Write header
            feature_names = ['person_name'] + [f'dlib_{i}' for i in range(dlib_features)]
            
            if self.pca is not None:
                feature_names += [f'cnn_pca_{i}' for i in range(cnn_features)]
            else:
                feature_names += [f'cnn_scaled_{i}' for i in range(cnn_features)]
            
            writer.writerow(feature_names)
            
            # Write features
            for person_name, features in zip(person_names, features_array):
                row = [person_name] + features.tolist()
                writer.writerow(row)
        
        self.logger.info(f"Features saved to: {self.features_file}")
    
    def run(self):
        """Main execution function"""
        self.logger.info("Starting enhanced face feature extraction...")
        
        # Get person folders
        person_folders = [f for f in os.listdir(self.path_images_from_camera) 
                         if os.path.isdir(os.path.join(self.path_images_from_camera, f))]
        person_folders.sort()
        
        if not person_folders:
            self.logger.error("No person folders found!")
            return
        
        self.logger.info(f"Found {len(person_folders)} person folders")
        
        # Extract features for all persons
        all_features = []
        person_names = []
        
        for person_folder in person_folders:
            self.logger.info(f"Processing person: {person_folder}")
            
            # Get person name
            if len(person_folder.split('_')) >= 3:
                person_name = person_folder.split('_', 2)[-1]
            else:
                person_name = person_folder
            
            # Extract features
            features = self.process_person_images(person_folder)
            
            all_features.append(features)
            person_names.append(person_name)
            
            self.logger.info(f"Completed processing: {person_folder}")
        
        # Convert to numpy array
        all_features = np.array(all_features)
        
        # Fit preprocessing components
        self.fit_preprocessing_components(all_features)
        
        # Process features
        processed_features = self.process_all_features(all_features)
        
        # Save to CSV
        self.save_features_to_csv(person_names, processed_features)
        
        self.logger.info("Enhanced face feature extraction completed successfully!")
        self.logger.info(f"Total persons processed: {len(person_names)}")
        
        # Log feature dimensions
        total_features = processed_features.shape[1]
        cnn_features = total_features - 128
        pca_status = "with PCA" if self.pca is not None else "without PCA (scaled only)"
        
        self.logger.info(f"Feature dimensions: {total_features} (128 Dlib + {cnn_features} CNN {pca_status})")

def main():
    """Main function"""
    extractor = EnhancedFaceFeatureExtractor()
    extractor.run()

if __name__ == '__main__':
    main()