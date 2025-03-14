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
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # 设置模型文件路径
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        # 状态变量
        self.current_image_path = None
        self.emoji_image = None
        
        # 创建界面
        self.create_widgets()
        
        # 加载检测模型
        self.load_detector()
    
    def load_detector(self):
        """加载人脸检测器和特征点预测器"""
        try:
            self.detector = dlib.get_frontal_face_detector()
            
            # 检查模型文件是否存在
            if not os.path.exists(self.predictor_path):
                self.show_model_download_instructions()
                return
                
            self.predictor = dlib.shape_predictor(self.predictor_path)
            self.update_status("Loaded the model, please upload...")
        except Exception as e:
            messagebox.showerror("error", f"Error when loading the picture: {str(e)}")
            self.update_status("filed to load the model")
    
    def show_model_download_instructions(self):
        """显示模型下载说明"""
        msg = (
            "未找到面部特征点模型文件！\n\n"
            "请下载 shape_predictor_68_face_landmarks.dat 文件:\n"
            "1. 访问: https://github.com/davisking/dlib-models\n"
            "2. 下载: shape_predictor_68_face_landmarks.dat.bz2\n"
            "3. 解压文件\n"
            "4. 将解压后的 .dat 文件放在程序同一目录下\n\n"
            "然后重新启动应用程序。"
        )
        messagebox.showwarning("缺少模型文件", msg)
        self.update_status("未找到模型文件，请下载后重启程序")
    
    def create_widgets(self):
        """创建应用界面部件"""
        # 主框架
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 上传按钮
        upload_btn = tk.Button(main_frame, text="Upload photo", command=self.upload_and_process,
                             bg="#4CAF50", fg="white", font=("Arial", 12), height=2)
        upload_btn.pack(fill=tk.X, padx=20, pady=10)
        
        # 图片区域框架
        images_frame = tk.Frame(main_frame, bg="#f0f0f0")
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 图片区域分为左右两部分
        # 左侧 - 原始照片
        self.original_frame = tk.LabelFrame(images_frame, text="Original Photo", bg="#f0f0f0")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_label = tk.Label(self.original_frame, bg="white", text="Please upload the photo")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧 - 生成的Emoji
        self.emoji_frame = tk.LabelFrame(images_frame, text="Generated Emoji", bg="#f0f0f0")
        self.emoji_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.emoji_label = tk.Label(self.emoji_frame, bg="white", text="Emoji shows here")
        self.emoji_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 保存按钮
        self.save_btn = tk.Button(main_frame, text="保存Emoji", command=self.save_emoji,
                               bg="#2196F3", fg="white", font=("Arial", 12), height=2, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, padx=20, pady=10)
        
        # 状态栏
        self.status_bar = tk.Label(self.root, text="准备就绪...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_and_process(self):
        """上传并处理照片"""
        file_path = filedialog.askopenfilename(
            title="Select the photo",
            filetypes=[("Image file", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                # 保存当前图片路径
                self.current_image_path = file_path
                self.display_original_image(file_path)
                self.update_status("Analyzing...")
                self.analyze_and_generate(file_path)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error when handling the image: {str(e)}")
                self.update_status("Error")
    
    def display_original_image(self, image_path):
        """显示原始照片"""
        try:
            # 打开图片并调整大小
            image = Image.open(image_path)
            image = self.resize_image(image, max_size=(350, 300))
            
            # 转换为PhotoImage对象
            photo = ImageTk.PhotoImage(image)
            
            # 更新标签
            self.original_label.config(image=photo)
            self.original_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"显示照片时出错: {str(e)}")
    
    def resize_image(self, image, max_size):
        """调整图片大小适应显示区域"""
        width, height = image.size
        max_width, max_height = max_size
        
        # 计算缩放比例
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    
    def analyze_and_generate(self, image_path):
        """分析人脸并生成Emoji"""
        try:
            # 读取图片并预处理
            image = self.preprocess_image(image_path)
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
              
            filtered_gray = cv2.bilateralFilter(gray, 9, 75, 75)

            # 检测人脸
            faces = self.detector(filtered_gray)
            
            if len(faces) == 0:
                messagebox.showwarning("Warning", "Didn't detect any face")
                self.update_status("No face detected")
                return
            
            # 使用最大的人脸
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # 获取面部特征点
            landmarks = self.predictor(gray, face)
            
            # 转换特征点为坐标
            points = []
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                points.append((x, y))
            
            # 计算嘴巴弧度
            mouth_curvature = self.calculate_mouth_curvature(points)
            
            # 计算眼睛闭合度
            eye_closure = self.calculate_eye_closure(points)
            
            # 创建特征字典
            features = {
                "mouth_curvature": mouth_curvature,
                "eye_closure": eye_closure
            }
            emotion = emotion_recognition(image_path)
            # 生成Emoji
            generator = EmojiGenerator(size=(350, 350),emotion=emotion)
            self.emoji_image = generator.generate_emoji(features)
            
            # 显示Emoji
            self.display_emoji(self.emoji_image)
            
            # 启用保存按钮
            self.save_btn.config(state=tk.NORMAL)
            
            # 更新状态
            self.update_status("Emoji生成成功")
            
        except Exception as e:
            messagebox.showerror("错误", f"分析生成时出错: {str(e)}")
            self.update_status("分析生成失败")
    
    def preprocess_image(self, image_path):
        """图像预处理"""
        # 读取图像
        image = cv2.imread(image_path)
        
        # 检查图像尺寸
        height, width = image.shape[:2]
        max_dimension = 1024
        
        if max(height, width) > max_dimension:
            scale_factor = max_dimension / max(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 降噪
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return image
    
    def calculate_mouth_curvature(self, points):
        """计算嘴巴弧度"""
        # 嘴角点
        left_corner = points[48]  # 左嘴角
        right_corner = points[54]  # 右嘴角
        
        # 唇中心点
        top_center = points[51]  # 上唇中心点
        bottom_center = points[57]  # 下唇中心点
        
        # 计算嘴角平均高度
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        
        # 计算嘴角相对于上唇中心的位置
        relative_position = corner_avg_y - top_center[1]
        
        # 归一化
        mouth_width = abs(right_corner[0] - left_corner[0])
        normalized_curvature = (relative_position / mouth_width * 100) if mouth_width > 0 else 0
        
        # 计算开口程度
        mouth_openness = abs(top_center[1] - bottom_center[1]) / mouth_width * 100 if mouth_width > 0 else 0
        
        return {
            "normalized_curvature": normalized_curvature,
            "openness": mouth_openness,
            "width": mouth_width
        }
    
    def calculate_eye_closure(self, points):
        """计算眼睛闭合度"""
        # 左眼点 (36-41)
        left_eye = points[36:42]
        
        # 右眼点 (42-47)
        right_eye = points[42:48]
        
        # 计算左右眼闭合度
        left_closure = self.calculate_single_eye_closure(left_eye)
        right_closure = self.calculate_single_eye_closure(right_eye)
        
        # 平均闭合度
        avg_closure = (left_closure + right_closure) / 2
        
        return {
            "average": avg_closure
        }
    
    def calculate_single_eye_closure(self, eye_points):
        """计算单个眼睛的闭合度"""
        # 上眼睑点和下眼睑点
        top_points = [eye_points[1], eye_points[2]]
        bottom_points = [eye_points[5], eye_points[4]]
        
        # 计算平均高度
        total_dist = 0
        for i in range(len(top_points)):
            dist = np.sqrt((top_points[i][0] - bottom_points[i][0])**2 + 
                          (top_points[i][1] - bottom_points[i][1])**2)
            total_dist += dist
        
        avg_height = total_dist / len(top_points) if top_points else 0
        
        # 计算眼睛宽度
        width = np.sqrt((eye_points[0][0] - eye_points[3][0])**2 + 
                       (eye_points[0][1] - eye_points[3][1])**2)
        
        # 计算高宽比
        aspect_ratio = avg_height / width if width > 0 else 0
        
        # 归一化
        normal_ratio = 0.25  # 正常睁眼比例
        normalized_closure = min(max(aspect_ratio / normal_ratio, 0), 1)
        
        return normalized_closure
    
    def display_emoji(self, emoji_image):
        """显示生成的Emoji"""
        try:
            # 转换为PhotoImage对象
            photo = ImageTk.PhotoImage(emoji_image)
            
            # 更新标签
            self.emoji_label.config(image=photo)
            self.emoji_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"显示Emoji时出错: {str(e)}")
    
    def save_emoji(self):
        """保存生成的Emoji"""
        if self.emoji_image is None:
            messagebox.showwarning("警告", "请先生成Emoji")
            return
        
        try:
            # 选择保存位置
            file_path = filedialog.asksaveasfilename(
                title="保存Emoji",
                defaultextension=".png",
                filetypes=[("PNG图片", "*.png"), ("所有文件", "*.*")]
            )
            
            if file_path:
                # 保存图片
                self.emoji_image.save(file_path)
                
                # 更新状态
                self.update_status(f"Emoji已保存到: {os.path.basename(file_path)}")
                
                # 显示成功消息
                messagebox.showinfo("成功", "Emoji已成功保存！")
        except Exception as e:
            messagebox.showerror("错误", f"保存Emoji时出错: {str(e)}")
    
    def update_status(self, message):
        """更新状态栏消息"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()


class EmojiGenerator:
    """Emoji 生成器 - 基于面部特征创建自定义 emoji"""
    
    def __init__(self, size=(512, 512),emotion='happy'):
        """初始化生成器
        
        Args:
            size: 输出图像大小 (宽度, 高度)
        """
        self.size = size
        
        # 颜色配置
        self.colors = {
            "face": (255,218,154),  # 黄色表情
            "outline": (0, 0, 0),   # 黑色轮廓
            "eyes": (255, 255, 255),  # 白色眼白
            "pupils": (0, 0, 0),    # 黑色瞳孔
            "mouth": (0, 0, 0),     # 黑色嘴巴
            "tongue": (255, 80, 80) # 粉红色舌头
        }

        if emotion=='sad':self.colors['face']=(211,238,255)
        elif emotion=='angry':self.colors['face']=(255,208,203)
        elif emotion=='surprise':self.colors['face']=(255,219,251)
        elif emotion=='neutral':self.colors['face']=(255,230,0)
        elif emotion=='disgust':self.colors['face']=(191,255,163)
        elif emotion=='fearful':self.colors['face']=(223,163,155)
        print(self.colors)
        print(emotion)
        
    
    def generate_emoji(self, features):
        """基于面部特征生成 emoji
        
        Args:
            features: 面部特征分析的结果
            
        Returns:
            PIL.Image: 生成的 emoji 图像
        """
        # 创建空白图像
        image = Image.new('RGB', self.size, color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 提取相关特征
        mouth_curve = features["mouth_curvature"]["normalized_curvature"]
        mouth_openness = features["mouth_curvature"]["openness"]
        eye_closure = features["eye_closure"]["average"]
        
        # 设定基本参数
        center_x, center_y = self.size[0] // 2, self.size[1] // 2
        radius = min(self.size) // 2 - 20
        
        # 绘制脸部（黄色圆形）
        self._draw_face(draw, center_x, center_y, radius)
        
        # 绘制眼睛
        eye_distance = radius * 0.6
        left_eye_x = center_x - eye_distance // 2
        right_eye_x = center_x + eye_distance // 2
        eyes_y = center_y - radius * 0.1
        
        self._draw_eyes(draw, left_eye_x, right_eye_x, eyes_y, eye_closure, radius)
        
        # 绘制嘴巴
        mouth_y = center_y + radius * 0.3
        self._draw_mouth(draw, center_x, mouth_y, radius, mouth_curve, mouth_openness)
        
        return image
    
    def _draw_face(self, draw, cx, cy, radius):
        """绘制脸部
        
        Args:
            draw: PIL.ImageDraw 对象
            cx, cy: 中心坐标
            radius: 半径
        """
        # 绘制填充圆形
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=self.colors["face"],
            outline=self.colors["outline"],
            width=3
        )
    
    def _draw_eyes(self, draw, left_x, right_x, y, closure_ratio, face_radius):
        """绘制眼睛
        
        Args:
            draw: PIL.ImageDraw 对象
            left_x, right_x: 左右眼x坐标
            y: 眼睛y坐标
            closure_ratio: 眼睛闭合度 (0-1)
            face_radius: 脸部半径（用于比例计算）
        """
        # 根据闭合度调整眼睛高度
        eye_width = face_radius * 0.2
        max_eye_height = face_radius * 0.15
        eye_height = max_eye_height * closure_ratio
        
        # 确保最小高度（即使闭眼也要有一条线）
        eye_height = max(eye_height, 2)
        
        # 左眼
        left_eye_bbox = [
            (left_x - eye_width//2, y - eye_height//2),
            (left_x + eye_width//2, y + eye_height//2)
        ]
        
        # 右眼
        right_eye_bbox = [
            (right_x - eye_width//2, y - eye_height//2),
            (right_x + eye_width//2, y + eye_height//2)
        ]
        
        # 绘制眼睛
        if closure_ratio < 0.2:  # 基本闭眼
            # 绘制曲线表示闭眼
            draw.line(
                [(left_eye_bbox[0][0], y), (left_eye_bbox[1][0], y)],
                fill=self.colors["outline"], width=3
            )
            draw.line(
                [(right_eye_bbox[0][0], y), (right_eye_bbox[1][0], y)],
                fill=self.colors["outline"], width=3
            )
        else:
            # 绘制椭圆形眼睛
            draw.ellipse(left_eye_bbox, fill=self.colors["eyes"], outline=self.colors["outline"], width=2)
            draw.ellipse(right_eye_bbox, fill=self.colors["eyes"], outline=self.colors["outline"], width=2)
            
            # 根据闭合度调整瞳孔大小
            pupil_size = min(eye_width, eye_height) * 0.4
            
            # 绘制瞳孔
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
    
    def _draw_mouth(self, draw, cx, cy, face_radius, curvature, openness):
        """绘制嘴巴
        
        Args:
            draw: PIL.ImageDraw 对象
            cx, cy: 中心坐标
            face_radius: 脸部半径
            curvature: 嘴巴弧度
            openness: 嘴巴开合度
        """
        # 调整曲率范围，映射到合适的控制点位置
        # 限制曲率范围避免过度变形
        curvature = max(min(curvature, 30), -30)
        
        # 设置嘴巴大小
        mouth_width = face_radius * 0.7
        
        # 嘴巴的端点
        left_x = cx - mouth_width // 2
        right_x = cx + mouth_width // 2
        
        # 计算控制点
        control_y_offset = curvature * face_radius / 100
        
        # 判断嘴巴类型
        if openness > 20:  # 张开的嘴巴
            # 上嘴唇
            upper_points = [
                (left_x, cy),
                (cx, cy - control_y_offset * 0.5),  # 上唇中点稍微上移
                (right_x, cy)
            ]
            
            # 计算下嘴唇的位置（根据开口度）
            mouth_height = openness * face_radius / 100
            lower_points = [
                (left_x, cy),
                (cx, cy + mouth_height),  # 下唇中点
                (right_x, cy)
            ]
            
            # 绘制上嘴唇
            draw.line(upper_points, fill=self.colors["mouth"], width=3, joint="curve")
            
            # 绘制下嘴唇
            draw.line(lower_points, fill=self.colors["mouth"], width=3, joint="curve")
            
            # 如果是张大嘴，画舌头
            if openness > 40:
                # 舌头区域
                tongue_width = mouth_width * 0.6
                tongue_height = mouth_height * 0.6
                tongue_bbox = [
                    (cx - tongue_width//2, cy),
                    (cx + tongue_width//2, cy + tongue_height)
                ]
                draw.ellipse(tongue_bbox, fill=self.colors["tongue"])
        else:
            # 闭合嘴巴（单曲线）
            points = [
                (left_x, cy),
                (cx, cy + control_y_offset),  # 控制点
                (right_x, cy)
            ]
            
            # 绘制单曲线
            draw.line(points, fill=self.colors["mouth"], width=4, joint="curve")
            
            # 对于微笑，添加第二条曲线增强效果
            if curvature > 10:
                smile_points = [
                    (left_x + mouth_width * 0.1, cy + 4),
                    (cx, cy + control_y_offset + 4),
                    (right_x - mouth_width * 0.1, cy + 4)
                ]
                draw.line(smile_points, fill=self.colors["mouth"], width=3, joint="curve")
    
    def save_emoji(self, emoji_image, output_path):
        """保存 emoji 图像
        
        Args:
            emoji_image: PIL.Image 对象
            output_path: 输出路径
        """
        emoji_image.save(output_path)
        return output_path



def emotion_recognition(image_path):
    emo = DeepFace.analyze(image_path, actions = ['emotion'],enforce_detection=False)
    dom=emo[0]['dominant_emotion']
    # print(emo)
    return dom

def main():
    root = tk.Tk()
    app = SimpleFaceEmojiApp(root)
    root.mainloop()
    


if __name__ == "__main__":
   
    main()