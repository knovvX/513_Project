import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import dlib
from math import atan2, degrees
from deepface import DeepFace
import argparse

class SimpleFaceEmojiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emoji Generator")
        self.root.geometry("1000x600")
        self.root.configure(bg="#f0f0f0")
        
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        self.current_image_path = None
        self.emoji_image = None
        self.dlib_analysis_image = None
        
        self.create_widgets()
        self.load_detector()
    
    def load_detector(self):
        try:
            self.detector = dlib.get_frontal_face_detector()
            
            if not os.path.exists(self.predictor_path):
                self.show_model_download_instructions()
                return
                
            self.predictor = dlib.shape_predictor(self.predictor_path)
            self.update_status("Model loaded, please upload a photo...")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading models: {str(e)}")
            self.update_status("Failed to load models")
    
    def show_model_download_instructions(self):
        msg = (
            "Face landmark model file not found!\n\n"
            "Please download the shape_predictor_68_face_landmarks.dat file:\n"
            "1. Visit: https://github.com/davisking/dlib-models\n"
            "2. Download: shape_predictor_68_face_landmarks.dat.bz2\n"
            "3. Extract the file\n"
            "4. Place the .dat file in the same directory as this program\n\n"
            "Then restart the application."
        )
        messagebox.showwarning("Missing Model File", msg)
        self.update_status("Model file not found, please download and restart")
    
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        upload_btn = tk.Button(main_frame, text="Upload Photo", command=self.upload_and_process,
                             bg="#4CAF50", fg="white", font=("Arial", 12), height=2)
        upload_btn.pack(fill=tk.X, padx=20, pady=10)
        
        images_frame = tk.Frame(main_frame, bg="#f0f0f0")
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.original_frame = tk.LabelFrame(images_frame, text="Original Photo", bg="#f0f0f0")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_label = tk.Label(self.original_frame, bg="white", text="Please upload a photo")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dlib_frame = tk.LabelFrame(images_frame, text="Dlib Analysis", bg="#f0f0f0")
        self.dlib_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dlib_label = tk.Label(self.dlib_frame, bg="white", text="Dlib analysis will show here")
        self.dlib_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.emoji_frame = tk.LabelFrame(images_frame, text="Generated Emoji", bg="#f0f0f0")
        self.emoji_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.emoji_label = tk.Label(self.emoji_frame, bg="white", text="Emoji will appear here")
        self.emoji_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.save_btn = tk.Button(main_frame, text="Save Emoji", command=self.save_emoji,
                               bg="#2196F3", fg="white", font=("Arial", 12), height=2, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_bar = tk.Label(self.root, text="Ready...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_and_process(self):
        file_path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.display_original_image(file_path)
                self.update_status("Analyzing photo...")
                self.analyze_and_generate(file_path)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
                self.update_status("Error processing image")
    
    def display_original_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = self.resize_image(image, max_size=(300, 250))
            
            photo = ImageTk.PhotoImage(image)
            
            self.original_label.config(image=photo)
            self.original_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying image: {str(e)}")
    
    def resize_image(self, image, max_size):
        width, height = image.size
        max_width, max_height = max_size
        
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    
    def analyze_and_generate(self, image_path):
        try:
            image = self.preprocess_image(image_path)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
              
            filtered_gray = cv2.bilateralFilter(gray, 9, 75, 75)

            faces = self.detector(filtered_gray)
            
            if len(faces) == 0:
                messagebox.showwarning("Warning", "No face detected")
                self.update_status("No face detected")
                return
            
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            landmarks = self.predictor(gray, face)
            
            points = []
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                points.append((x, y))
            
            self.create_dlib_analysis_image(image.copy(), face, points)
            
            mouth_curvature = self.calculate_mouth_curvature(points)
            
            eye_closure = self.calculate_eye_closure(points)
            
            features = {
                "mouth_curvature": mouth_curvature,
                "eye_closure": eye_closure
            }
            
            emotion = emotion_recognition(image_path)
            
            generator = EmojiGenerator(size=(300, 300), emotion=emotion)
            self.emoji_image = generator.generate_emoji(features)
            
            self.display_emoji(self.emoji_image)
            
            self.save_btn.config(state=tk.NORMAL)
            
            self.update_status("Emoji generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating emoji: {str(e)}")
            self.update_status("Error generating emoji")
    
    def create_dlib_analysis_image(self, image, face, points):
        try:
            cv2.rectangle(image, 
                         (face.left(), face.top()), 
                         (face.right(), face.bottom()),
                         (0, 255, 0), 2)
            
            for point in points:
                cv2.circle(image, point, 2, (0, 0, 255), -1)
            
            for i in range(17, 21):
                cv2.line(image, points[i], points[i+1], (255, 0, 0), 1)
            for i in range(22, 26):
                cv2.line(image, points[i], points[i+1], (255, 0, 0), 1)
                
            for i in range(27, 30):
                cv2.line(image, points[i], points[i+1], (255, 0, 0), 1)
            for i in [30, 31, 33, 34]:
                cv2.line(image, points[i], points[i+1], (255, 0, 0), 1)
                
            for i in range(36, 41):
                cv2.line(image, points[i], points[i+1], (255, 0, 0), 1)
            cv2.line(image, points[41], points[36], (255, 0, 0), 1)
            
            for i in range(42, 47):
                cv2.line(image, points[i], points[i+1], (255, 0, 0), 1)
            cv2.line(image, points[47], points[42], (255, 0, 0), 1)
            
            for i in range(48, 59):
                cv2.line(image, points[i], points[i+1], (255, 0, 0), 1)
            cv2.line(image, points[59], points[48], (255, 0, 0), 1)
            
            for i in range(60, 67):
                cv2.line(image, points[i], points[i+1], (255, 0, 0), 1)
            cv2.line(image, points[67], points[60], (255, 0, 0), 1)
            
            cv2.putText(image, "Face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            pil_image = self.resize_image(pil_image, max_size=(300, 250))
            
            self.dlib_analysis_image = pil_image
            
            photo = ImageTk.PhotoImage(pil_image)
            self.dlib_label.config(image=photo)
            self.dlib_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creating analysis image: {str(e)}")
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        
        height, width = image.shape[:2]
        max_dimension = 1024
        
        if max(height, width) > max_dimension:
            scale_factor = max_dimension / max(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image
    
    def calculate_mouth_curvature(self, points):
        left_corner = points[48]
        right_corner = points[54]
        
        top_center = points[51]
        bottom_center = points[57]
        
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        
        relative_position = corner_avg_y - top_center[1]
        
        mouth_width = abs(right_corner[0] - left_corner[0])
        normalized_curvature = (relative_position / mouth_width * 100) if mouth_width > 0 else 0
        
        mouth_openness = abs(top_center[1] - bottom_center[1]) / mouth_width * 100 if mouth_width > 0 else 0
        
        return {
            "normalized_curvature": normalized_curvature,
            "openness": mouth_openness,
            "width": mouth_width
        }
    
    def calculate_eye_closure(self, points):
        left_eye = points[36:42]
        
        right_eye = points[42:48]
        
        left_closure = self.calculate_single_eye_closure(left_eye)
        right_closure = self.calculate_single_eye_closure(right_eye)
        
        avg_closure = (left_closure + right_closure) / 2
        
        return {
            "average": avg_closure
        }
    
    def calculate_single_eye_closure(self, eye_points):
        top_points = [eye_points[1], eye_points[2]]
        bottom_points = [eye_points[5], eye_points[4]]
        
        total_dist = 0
        for i in range(len(top_points)):
            dist = np.sqrt((top_points[i][0] - bottom_points[i][0])**2 + 
                          (top_points[i][1] - bottom_points[i][1])**2)
            total_dist += dist
        
        avg_height = total_dist / len(top_points) if top_points else 0
        
        width = np.sqrt((eye_points[0][0] - eye_points[3][0])**2 + 
                       (eye_points[0][1] - eye_points[3][1])**2)
        
        aspect_ratio = avg_height / width if width > 0 else 0
        
        normal_ratio = 0.25
        normalized_closure = min(max(aspect_ratio / normal_ratio, 0), 1)
        
        return normalized_closure
    
    def display_emoji(self, emoji_image):
        try:
            photo = ImageTk.PhotoImage(emoji_image)
            
            self.emoji_label.config(image=photo)
            self.emoji_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying emoji: {str(e)}")
    
    def save_emoji(self):
        if self.emoji_image is None:
            messagebox.showwarning("Warning", "Please generate an emoji first")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Emoji",
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")]
            )
            
            if file_path:
                self.emoji_image.save(file_path)
                
                self.update_status(f"Emoji saved to: {os.path.basename(file_path)}")
                
                messagebox.showinfo("Success", "Emoji saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving emoji: {str(e)}")
    
    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()


class EmojiGenerator:
    def __init__(self, size=(512, 512), emotion='happy'):
        self.size = size
        
        self.colors = {
            "face": (255,218,154),
            "outline": (0, 0, 0),
            "eyes": (255, 255, 255),
            "pupils": (0, 0, 0),
            "mouth": (0, 0, 0),
            "tongue": (255, 80, 80)
        }

        if emotion=='sad':
            self.colors['face']=(211,238,255)
        elif emotion=='angry':
            self.colors['face']=(255,208,203)
        elif emotion=='surprise':
            self.colors['face']=(255,219,251)
        elif emotion=='neutral':
            self.colors['face']=(255,230,0)
        elif emotion=='disgust':
            self.colors['face']=(191,255,163)
        elif emotion=='fear':
            self.colors['face']=(223,163,155)
        
        print(f"Emotion detected: {emotion}")
        print(f"Face color set to: {self.colors['face']}")
    
    def generate_emoji(self, features):
        image = Image.new('RGB', self.size, color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        mouth_curve = features["mouth_curvature"]["normalized_curvature"]
        mouth_openness = features["mouth_curvature"]["openness"]
        eye_closure = features["eye_closure"]["average"]
        
        center_x, center_y = self.size[0] // 2, self.size[1] // 2
        radius = min(self.size) // 2 - 20
        
        self._draw_face(draw, center_x, center_y, radius)
        
        eye_distance = radius * 0.6
        left_eye_x = center_x - eye_distance // 2
        right_eye_x = center_x + eye_distance // 2
        eyes_y = center_y - radius * 0.1
        
        self._draw_eyes(draw, left_eye_x, right_eye_x, eyes_y, eye_closure, radius)
        
        mouth_y = center_y + radius * 0.3
        self._draw_mouth(draw, center_x, mouth_y, radius, mouth_curve, mouth_openness)
        
        return image
    
    def _draw_face(self, draw, cx, cy, radius):
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=self.colors["face"],
            outline=self.colors["outline"],
            width=3
        )
    
    def _draw_eyes(self, draw, left_x, right_x, y, closure_ratio, face_radius):
        eye_width = face_radius * 0.2
        max_eye_height = face_radius * 0.15
        eye_height = max_eye_height * closure_ratio
        
        eye_height = max(eye_height, 2)
        
        left_eye_bbox = [
            (left_x - eye_width//2, y - eye_height//2),
            (left_x + eye_width//2, y + eye_height//2)
        ]
        
        right_eye_bbox = [
            (right_x - eye_width//2, y - eye_height//2),
            (right_x + eye_width//2, y + eye_height//2)
        ]
        
        if closure_ratio < 0.2:
            draw.line(
                [(left_eye_bbox[0][0], y), (left_eye_bbox[1][0], y)],
                fill=self.colors["outline"], width=3
            )
            draw.line(
                [(right_eye_bbox[0][0], y), (right_eye_bbox[1][0], y)],
                fill=self.colors["outline"], width=3
            )
        else:
            draw.ellipse(left_eye_bbox, fill=self.colors["eyes"], outline=self.colors["outline"], width=2)
            draw.ellipse(right_eye_bbox, fill=self.colors["eyes"], outline=self.colors["outline"], width=2)
            
            pupil_size = min(eye_width, eye_height) * 0.4
            
            draw.ellipse(
                [(left_x - pupil_size//2, y - pupil_size//2),
                 (left_x + pupil_size//2, y + pupil_size//2)],
                fill=self.colors["pupils"]
            )
            draw.ellipse(
                [(right_x - pupil_size//2, y - pupil_size//2),
                 (right_x + pupil_size//2, y + pupil_size//2)],
                fill=self.colors["pupils"]
            )
    
    # def _draw_mouth(self, draw, cx, cy, face_radius, curvature, openness):
        # curvature = max(min(curvature, 30), -30)
        
        # mouth_width = face_radius * 0.7
        
        # left_x = cx - mouth_width // 2
        # right_x = cx + mouth_width // 2
        
        # control_y_offset = curvature * face_radius / 100
        
        # if openness > 20:
        #     upper_points = [
        #         (left_x, cy),
        #         (cx, cy - control_y_offset * 0.5),
        #         (right_x, cy)
        #     ]
            
        #     mouth_height = openness * face_radius / 100
        #     lower_points = [
        #         (left_x, cy),
        #         (cx, cy + mouth_height),
        #         (right_x, cy)
        #     ]
            
        #     draw.line(upper_points, fill=self.colors["mouth"], width=3, joint="curve")
            
        #     draw.line(lower_points, fill=self.colors["mouth"], width=3, joint="curve")
            
        #     if openness > 40:
        #         tongue_width = mouth_width * 0.6
        #         tongue_height = mouth_height * 0.6
        #         tongue_bbox = [
        #             (cx - tongue_width//2, cy),
        #             (cx + tongue_width//2, cy + tongue_height)
        #         ]
        #         draw.ellipse(tongue_bbox, fill=self.colors["tongue"])
        # else:
        #     points = [
        #         (left_x, cy),
        #         (cx, cy + control_y_offset),
        #         (right_x, cy)
        #     ]
            
        #     draw.line(points, fill=self.colors["mouth"], width=4, joint="curve")
            
        #     if curvature > 10:
        #         smile_points = [
        #             (left_x + mouth_width * 0.1, cy + 4),
        #             (cx, cy + control_y_offset + 4),
        #             (right_x - mouth_width * 0.1, cy + 4)
        #         ]
        #         draw.line(smile_points, fill=self.colors["mouth"], width=3, joint="curve")
    def _draw_mouth(self, draw, cx, cy, face_radius, curvature, openness):
        # 限制曲率和开合度到合理范围
        curvature = max(min(curvature, 30), -30)
        
        # 调整嘴巴宽度
        mouth_width = face_radius * 0.6
        
        # 嘴巴的端点
        left_x = cx - mouth_width // 2
        right_x = cx + mouth_width // 2
        
        # 根据曲率计算控制点偏移
        control_y_offset = curvature * face_radius / 100
        
        # 获取一个更合理的开合度值
        adjusted_openness = min(openness * 0.7, 40)
        
        if adjusted_openness > 15:  # 张开的嘴巴
            # 计算嘴巴高度
            mouth_height = adjusted_openness * face_radius / 120
            
            # 上唇和下唇的中心点位置
            upper_lip_y = cy - mouth_height * 0.2
            lower_lip_y = cy + mouth_height * 0.8
            
            # 创建嘴巴形状 - 使用椭圆和多边形组合
            
            # 绘制嘴巴轮廓 - 外部圆形
            outer_bbox = [
                (left_x, upper_lip_y),
                (right_x, lower_lip_y)
            ]
            # 只绘制椭圆的上半部分和下半部分，中间留空
            draw.arc(outer_bbox, 0, 180, fill=self.colors["mouth"], width=3)  # 上半部分
            draw.arc(outer_bbox, 180, 360, fill=self.colors["mouth"], width=3)  # 下半部分
            
            # 绘制左右两侧连接线
            draw.line([(left_x, upper_lip_y), (left_x, lower_lip_y)], 
                    fill=self.colors["mouth"], width=3)
            draw.line([(right_x, upper_lip_y), (right_x, lower_lip_y)], 
                    fill=self.colors["mouth"], width=3)
            
            # 调整内部区域 - 嘴内形状
            inner_padding = mouth_width * 0.08
            inner_mouth = [
                (left_x + inner_padding, upper_lip_y + inner_padding),
                (right_x - inner_padding, lower_lip_y - inner_padding)
            ]
            
            # 填充嘴内深色区域
            draw.rectangle(inner_mouth, fill=(50, 50, 50))
            
            # 对于张大的嘴巴，添加舌头
            if adjusted_openness > 25:
                tongue_width = mouth_width * 0.5
                tongue_height = mouth_height * 0.5
                
                # 舌头以嘴巴中心为基准，位于嘴巴下半部分
                tongue_top = cy + inner_padding
                tongue_left = cx - tongue_width // 2
                tongue_right = cx + tongue_width // 2
                tongue_bottom = lower_lip_y - inner_padding
                
                # 创建舌头形状 - 使用半圆弧加上矩形
                # 舌头的上部是半圆形
                draw.chord(
                    [(tongue_left, tongue_top), (tongue_right, tongue_bottom)],
                    180, 0, fill=self.colors["tongue"]
                )
        
        else:  # 闭合的嘴巴
            # 闭合嘴巴只需要一条曲线/弧线
            if curvature >= 0:  # 微笑或中性
                # 计算控制点 - 为了微笑效果
                control_y = cy + control_y_offset
                
                # 创建微笑曲线 - 使用多个点
                points = [
                    left_x, cy,
                    left_x + mouth_width * 0.25, control_y - mouth_width * 0.05,
                    cx, control_y,
                    right_x - mouth_width * 0.25, control_y - mouth_width * 0.05,
                    right_x, cy
                ]
                
                # 绘制更平滑的曲线
                draw.line([(left_x, cy), (right_x, cy)], fill=self.colors["mouth"], width=1)
                draw.line(points, fill=self.colors["mouth"], width=3, joint="curve")
                
                # 如果是非常开心，添加更深的微笑线
                if curvature > 15:
                    second_control_y = control_y + 4
                    second_points = [
                        left_x + mouth_width * 0.1, cy + 2,
                        cx - mouth_width * 0.2, second_control_y,
                        cx, second_control_y + 2,
                        cx + mouth_width * 0.2, second_control_y,
                        right_x - mouth_width * 0.1, cy + 2
                    ]
                    draw.line(second_points, fill=self.colors["mouth"], width=2, joint="curve")
            
            else:  # 皱眉/悲伤
                # 计算控制点 - 为了皱眉效果
                control_y = cy + control_y_offset
                
                # 创建皱眉曲线
                points = [
                    left_x, cy,
                    left_x + mouth_width * 0.25, control_y + mouth_width * 0.05,
                    cx, control_y,
                    right_x - mouth_width * 0.25, control_y + mouth_width * 0.05,
                    right_x, cy
                ]
                
                # 绘制更平滑的曲线
                draw.line(points, fill=self.colors["mouth"], width=3, joint="curve")
    def save_emoji(self, emoji_image, output_path):
        emoji_image.save(output_path)
        return output_path


def emotion_recognition(image_path):
    emo = DeepFace.analyze(image_path, actions = ['emotion'], enforce_detection=False)
    dom = emo[0]['dominant_emotion']
    return dom

def main():
    root = tk.Tk()
    app = SimpleFaceEmojiApp(root)
    root.mainloop()
    

if __name__ == "__main__":
    main()