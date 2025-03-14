import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import dlib
import requests
from io import BytesIO
import hashlib
import json

class MemojiGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Memoji Generator")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # API configuration
        self.api_base_url = "https://api.apback.com/memoji"  # Replace with actual API URL if different
        self.api_key = None  # You'll need to replace this with your actual API key
        
        # Face detection model path
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        # State variables
        self.current_image_path = None
        self.memoji_image = None
        self.face_features = None
        
        # Create UI
        self.create_widgets()
        
        # Load face detector
        self.load_detector()
    
    def load_detector(self):
        """Load face detector and landmarks predictor"""
        try:
            self.detector = dlib.get_frontal_face_detector()
            
            if not os.path.exists(self.predictor_path):
                self.show_model_download_instructions()
                return
                
            self.predictor = dlib.shape_predictor(self.predictor_path)
            self.update_status("Model loaded. Ready to upload photo...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load face detection model: {str(e)}")
            self.update_status("Model loading failed.")
    
    def show_model_download_instructions(self):
        """Show instructions for downloading the face landmark model"""
        msg = (
            "Face landmark model file not found!\n\n"
            "Please download shape_predictor_68_face_landmarks.dat file:\n"
            "1. Visit: https://github.com/davisking/dlib-models\n"
            "2. Download: shape_predictor_68_face_landmarks.dat.bz2\n"
            "3. Extract the file\n"
            "4. Place the .dat file in the same directory as this program\n\n"
            "Then restart the application."
        )
        messagebox.showwarning("Missing Model File", msg)
        self.update_status("Missing model file.")
    
    def create_widgets(self):
        """Create application UI"""
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # API Key entry
        api_frame = tk.Frame(main_frame, bg="#f0f0f0")
        api_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(api_frame, text="API Key:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.api_key_entry = tk.Entry(api_frame, width=40, show="*")
        self.api_key_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Upload button
        self.upload_btn = tk.Button(
            main_frame, 
            text="Upload Photo", 
            command=self.upload_and_process,
            bg="#4CAF50", 
            fg="white", 
            font=("Arial", 12), 
            height=2
        )
        self.upload_btn.pack(fill=tk.X, padx=20, pady=10)
        
        # Images area
        images_frame = tk.Frame(main_frame, bg="#f0f0f0")
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Original photo
        self.original_frame = tk.LabelFrame(images_frame, text="Original Photo", bg="#f0f0f0")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_label = tk.Label(self.original_frame, bg="white", text="Please upload a photo")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side - Generated Memoji
        self.memoji_frame = tk.LabelFrame(images_frame, text="Generated Memoji", bg="#f0f0f0")
        self.memoji_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.memoji_label = tk.Label(self.memoji_frame, bg="white", text="Memoji will appear here")
        self.memoji_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Save button
        self.save_btn = tk.Button(
            main_frame, 
            text="Save Memoji", 
            command=self.save_memoji,
            bg="#2196F3", 
            fg="white", 
            font=("Arial", 12), 
            height=2, 
            state=tk.DISABLED
        )
        self.save_btn.pack(fill=tk.X, padx=20, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_and_process(self):
        """Upload and process a photo"""
        # Check if API key is provided
        self.api_key = self.api_key_entry.get().strip()
        if not self.api_key:
            messagebox.showwarning("Warning", "Please enter your API key first.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # Save current image path
                self.current_image_path = file_path
                
                # Display original photo
                self.display_original_image(file_path)
                
                # Update status
                self.update_status("Analyzing face...")
                
                # Analyze face and generate Memoji
                self.analyze_face_and_generate_memoji(file_path)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process photo: {str(e)}")
                self.update_status("Processing failed.")
    
    def display_original_image(self, image_path):
        """Display original photo"""
        try:
            # Open and resize image
            image = Image.open(image_path)
            image = self.resize_image(image, max_size=(350, 300))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.original_label.config(image=photo)
            self.original_label.image = photo  # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def resize_image(self, image, max_size):
        """Resize image to fit display area"""
        width, height = image.size
        max_width, max_height = max_size
        
        # Calculate scale ratio
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    
    def analyze_face_and_generate_memoji(self, image_path):
        """Analyze face features and generate Memoji"""
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Detect faces
            faces = self.detector(gray)
            
            if len(faces) == 0:
                messagebox.showwarning("Warning", "No face detected. Please try a clearer photo.")
                self.update_status("No face detected.")
                return
            
            # Use largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            
            # Convert landmarks to coordinates
            points = []
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                points.append((x, y))
            
            # Extract facial features
            self.face_features = self.extract_face_features(points, face)
            
            # Generate feature string for API
            feature_string = self.create_feature_string(self.face_features)
            
            # Call API to generate Memoji
            self.update_status("Generating Memoji...")
            self.generate_memoji_from_api(feature_string)
            
        except Exception as e:
            messagebox.showerror("Error", f"Face analysis failed: {str(e)}")
            self.update_status("Analysis failed.")
    
    def preprocess_image(self, image_path):
        """Preprocess image for face detection"""
        # Read image
        image = cv2.imread(image_path)
        
        # Check image size
        height, width = image.shape[:2]
        max_dimension = 1024
        
        if max(height, width) > max_dimension:
            scale_factor = max_dimension / max(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image
    
    def extract_face_features(self, points, face):
        """Extract key facial features from landmarks"""
        # Extract face shape
        jaw_points = points[0:17]  # Jaw line points
        
        # Extract eyes features
        left_eye = points[36:42]
        right_eye = points[42:48]
        left_eyebrow = points[17:22]
        right_eyebrow = points[22:27]
        
        # Calculate eye closure
        left_eye_closure = self.calculate_eye_closure(left_eye)
        right_eye_closure = self.calculate_eye_closure(right_eye)
        
        # Extract nose features
        nose_bridge = points[27:31]
        nose_tip = points[31:36]
        
        # Extract mouth features
        outer_lips = points[48:60]
        inner_lips = points[60:68]
        
        # Calculate mouth curvature
        mouth_curvature = self.calculate_mouth_curvature(outer_lips)
        
        # Determine face width and height
        face_width = face.width()
        face_height = face.height()
        
        # Create feature dictionary
        features = {
            "face_shape": {
                "width": face_width,
                "height": face_height,
                "ratio": face_width / face_height if face_height > 0 else 0,
                "jaw_line": jaw_points
            },
            "eyes": {
                "left_closure": left_eye_closure,
                "right_closure": right_eye_closure,
                "left_eyebrow_height": self.calculate_eyebrow_height(left_eyebrow, left_eye),
                "right_eyebrow_height": self.calculate_eyebrow_height(right_eyebrow, right_eye),
            },
            "nose": {
                "width": self.calculate_nose_width(nose_tip),
                "length": self.calculate_nose_length(nose_bridge, nose_tip),
            },
            "mouth": {
                "curvature": mouth_curvature,
                "openness": self.calculate_mouth_openness(inner_lips),
                "width": self.calculate_mouth_width(outer_lips),
            }
        }
        
        return features
    
    def calculate_eye_closure(self, eye_points):
        """Calculate eye closure ratio"""
        # Top and bottom eyelid points
        top_points = [eye_points[1], eye_points[2]]
        bottom_points = [eye_points[5], eye_points[4]]
        
        # Calculate average height
        total_dist = 0
        for i in range(len(top_points)):
            dist = np.sqrt((top_points[i][0] - bottom_points[i][0])**2 + 
                          (top_points[i][1] - bottom_points[i][1])**2)
            total_dist += dist
        
        avg_height = total_dist / len(top_points) if top_points else 0
        
        # Calculate eye width
        width = np.sqrt((eye_points[0][0] - eye_points[3][0])**2 + 
                      (eye_points[0][1] - eye_points[3][1])**2)
        
        # Calculate aspect ratio
        aspect_ratio = avg_height / width if width > 0 else 0
        
        return aspect_ratio
    
    def calculate_eyebrow_height(self, eyebrow_points, eye_points):
        """Calculate eyebrow height relative to eye"""
        # Average eyebrow y-position
        eyebrow_y = sum(p[1] for p in eyebrow_points) / len(eyebrow_points)
        
        # Average eye y-position
        eye_y = sum(p[1] for p in eye_points) / len(eye_points)
        
        # Distance between eyebrow and eye
        return eye_y - eyebrow_y
    
    def calculate_nose_width(self, nose_tip):
        """Calculate nose width"""
        return abs(nose_tip[0][0] - nose_tip[4][0])
    
    def calculate_nose_length(self, nose_bridge, nose_tip):
        """Calculate nose length"""
        # Bridge top to nose tip
        bridge_top = nose_bridge[0]
        tip = nose_tip[2]  # Center point of nose tip
        
        return abs(tip[1] - bridge_top[1])
    
    def calculate_mouth_curvature(self, mouth_points):
        """Calculate mouth curvature"""
        # Corner points
        left_corner = mouth_points[0]
        right_corner = mouth_points[6]
        
        # Top center point
        top_center = mouth_points[3]
        
        # Calculate average corner height
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        
        # Calculate relative position
        relative_position = corner_avg_y - top_center[1]
        
        # Normalize
        mouth_width = abs(right_corner[0] - left_corner[0])
        normalized_curvature = (relative_position / mouth_width) if mouth_width > 0 else 0
        
        return normalized_curvature
    
    def calculate_mouth_openness(self, inner_lips):
        """Calculate mouth openness"""
        # Top and bottom center points
        top_center = inner_lips[3]
        bottom_center = inner_lips[1]
        
        # Vertical distance
        return abs(top_center[1] - bottom_center[1])
    
    def calculate_mouth_width(self, outer_lips):
        """Calculate mouth width"""
        # Corner points
        left_corner = outer_lips[0]
        right_corner = outer_lips[6]
        
        return abs(right_corner[0] - left_corner[0])
    
    def create_feature_string(self, features):
        """Create a consistent string representation of features for API"""
        # Simplified feature extraction for API
        # Adjust according to what the API expects
        feature_dict = {
            "face_shape": "round" if features["face_shape"]["ratio"] > 0.85 else "oval",
            "eye_closure": (features["eyes"]["left_closure"] + features["eyes"]["right_closure"]) / 2,
            "eyebrow_height": (features["eyes"]["left_eyebrow_height"] + features["eyes"]["right_eyebrow_height"]) / 2,
            "nose_width": features["nose"]["width"],
            "mouth_curvature": features["mouth"]["curvature"],
            "mouth_openness": features["mouth"]["openness"],
        }
        
        # Convert to JSON string for hashing
        feature_json = json.dumps(feature_dict, sort_keys=True)
        
        # Create a hash to use as a consistent identifier
        feature_hash = hashlib.md5(feature_json.encode()).hexdigest()
        
        return feature_hash
    
    def generate_memoji_from_api(self, feature_string):
        """Call Apback Memoji API to generate Memoji"""
        try:
            # Construct API URL
            url = f"{self.api_base_url}/generate"
            
            # API parameters
            params = {
                "seed": feature_string,
                "type": "avatar",  # Adjust based on API documentation
                "format": "png"
            }
            
            # API headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "image/png"
            }
            
            # Make API request
            response = requests.get(url, params=params, headers=headers)
            
            # Check if request was successful
            if response.status_code == 200:
                # Load image from response
                self.memoji_image = Image.open(BytesIO(response.content))
                
                # Display Memoji
                self.display_memoji(self.memoji_image)
                
                # Enable save button
                self.save_btn.config(state=tk.NORMAL)
                
                # Update status
                self.update_status("Memoji generated successfully.")
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                messagebox.showerror("API Error", error_msg)
                self.update_status(f"API error: {response.status_code}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate Memoji: {str(e)}")
            self.update_status("Memoji generation failed.")
    
    def display_memoji(self, memoji_image):
        """Display generated Memoji"""
        try:
            # Resize image for display
            resized_image = self.resize_image(memoji_image, max_size=(350, 300))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(resized_image)
            
            # Update label
            self.memoji_label.config(image=photo)
            self.memoji_label.image = photo  # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display Memoji: {str(e)}")
    
    def save_memoji(self):
        """Save generated Memoji"""
        if self.memoji_image is None:
            messagebox.showwarning("Warning", "Please generate a Memoji first.")
            return
        
        try:
            # Choose save location
            file_path = filedialog.asksaveasfilename(
                title="Save Memoji",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
            )
            
            if file_path:
                # Save image
                self.memoji_image.save(file_path)
                
                # Update status
                self.update_status(f"Memoji saved to: {os.path.basename(file_path)}")
                
                # Show success message
                messagebox.showinfo("Success", "Memoji saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save Memoji: {str(e)}")
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    app = MemojiGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()