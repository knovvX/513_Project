import cv2
import numpy as np
import dlib
from PIL import Image, ImageDraw
import os
import argparse
from math import atan2, degrees
from deepface import DeepFace

class FacialFeatureAnalyzer:
    """人脸特征分析器 - 分析人脸并计算关键特征指标"""
    
    def __init__(self, predictor_path):
        """初始化分析器
        
        Args:
            predictor_path: dlib面部特征点预测模型的路径
        """
        # 加载dlib模型
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
    
    def analyze_image(self, image_path):
        """分析图像并提取面部特征
        
        Args:
            image_path: 输入图像的路径
            
        Returns:
            dict: 包含面部特征分析结果的字典
            str: 结果消息
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None, "无法读取图像文件"
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.detector(gray)
        if len(faces) == 0:
            return None, "未检测到人脸"
        
        # 使用最大的人脸（假设主体是最大的脸）
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # 检测面部特征点
        landmarks = self.predictor(gray, face)
        
        # 转换特征点为坐标数组
        points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append((x, y))
        
        # 计算关键面部特征
        features = {
            "mouth_curvature": self._calculate_mouth_curvature(points),
            "eye_closure": self._calculate_eye_closure(points),
            "face_width": face.width(),
            "face_height": face.height(),
            "face_position": (face.left(), face.top(), face.right(), face.bottom()),
            "landmarks": points
        }
        
        # 输出调试图像（可选）
        debug_image = image.copy()
        self._draw_debug_info(debug_image, points, features)
        cv2.imwrite("debug_analysis.jpg", debug_image)
        
        return features, "成功分析人脸特征"
    
    def _calculate_mouth_curvature(self, points):
        """计算嘴巴弧度
        
        Returns:
            dict: 包含多种嘴巴弧度测量值
        """
        # 嘴巴特征点索引
        # 48-54: 外唇轮廓
        mouth_outline = points[48:55]
        # 60-64: 内唇上部
        inner_top = points[60:65]
        # 64-67,60: 内唇下部
        inner_bottom = points[64:68]
        
        # 嘴角点
        left_corner = points[48]  # 左嘴角
        right_corner = points[54]  # 右嘴角
        
        # 唇中心点
        top_center = points[51]      # 上唇中心点
        bottom_center = points[57]   # 下唇中心点
        print(top_center)
        
        # 计算嘴角的平均高度
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        
        # 计算嘴角相对于上唇中心的位置
        # 正值表示微笑（嘴角上扬），负值表示不高兴（嘴角下垂）
        relative_position = corner_avg_y - top_center[1]
        
        # 归一化，根据嘴的宽度
        mouth_width = abs(right_corner[0] - left_corner[0])
        if mouth_width > 0:
            normalized_curvature = (relative_position / mouth_width) * 100
        else:
            normalized_curvature = 0
        
        # 计算开口程度
        mouth_openness = abs(top_center[1] - bottom_center[1]) / mouth_width
        
        # 计算嘴角角度
        left_angle = self._calculate_angle(left_corner, top_center)
        right_angle = self._calculate_angle(right_corner, top_center)
        
        return {
            "curvature": relative_position,
            "normalized_curvature": normalized_curvature,
            "openness": mouth_openness * 100,  # 百分比形式
            "left_angle": left_angle,
            "right_angle": right_angle,
            "width": mouth_width
        }
    
    def _calculate_eye_closure(self, points):
        """计算眼睛闭合度
        
        Returns:
            dict: 包含左眼和右眼的闭合度
        """
        # 眼睛特征点索引
        # 左眼: 36-41
        left_eye = points[36:42]
        # 右眼: 42-47
        right_eye = points[42:48]
        
        # 计算左眼闭合度
        left_eye_closure = self._calculate_single_eye_closure(left_eye)
        
        # 计算右眼闭合度
        right_eye_closure = self._calculate_single_eye_closure(right_eye)
        
        # 计算平均闭合度
        avg_closure = (left_eye_closure + right_eye_closure) / 2
        
        return {
            "left": left_eye_closure,
            "right": right_eye_closure,
            "average": avg_closure
        }
    
    def _calculate_single_eye_closure(self, eye_points):
        """计算单个眼睛的闭合度
        
        Args:
            eye_points: 眼睛的特征点
            
        Returns:
            float: 眼睛的闭合度 (0-1, 0 = 完全闭合, 1 = 完全睁开)
        """
        # 计算眼睛的高度（上下眼睑间距）
        top_points = [eye_points[1], eye_points[2]]  # 上眼睑
        bottom_points = [eye_points[5], eye_points[4]]  # 下眼睑
        
        # 计算上下眼睑之间的平均距离
        total_dist = 0
        for i in range(len(top_points)):
            dist = np.sqrt((top_points[i][0] - bottom_points[i][0])**2 + 
                         (top_points[i][1] - bottom_points[i][1])**2)
            total_dist += dist
        
        avg_height = total_dist / len(top_points)
        
        # 计算眼睛的宽度
        width = np.sqrt((eye_points[0][0] - eye_points[3][0])**2 + 
                       (eye_points[0][1] - eye_points[3][1])**2)
        
        # 计算高宽比
        if width > 0:
            aspect_ratio = avg_height / width
        else:
            aspect_ratio = 0
        
        # 归一化 (调整范围以更准确地反映眼睛状态)
        # 典型的正常睁眼 aspect_ratio 约为 0.2-0.3
        normal_ratio = 0.25  # 典型正常睁眼比例
        normalized_closure = min(max(aspect_ratio / normal_ratio, 0), 1)
        
        return normalized_closure
    
    def _calculate_angle(self, point1, point2):
        """计算两点之间连线的角度
        
        Args:
            point1: 第一个点的坐标 (x, y)
            point2: 第二个点的坐标 (x, y)
            
        Returns:
            float: 角度（度）
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return degrees(atan2(dy, dx))
    
    def _draw_debug_info(self, image, landmarks, features):
        """在图像上绘制调试信息
        
        Args:
            image: 输入图像
            landmarks: 面部特征点
            features: 计算出的特征
        """
        # 绘制所有特征点
        for i, (x, y) in enumerate(landmarks):
            # 特别标记嘴巴和眼睛点
            if 36 <= i <= 47:  # 眼睛
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            elif 48 <= i <= 67:  # 嘴巴
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            else:
                cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        
        # 添加嘴巴弧度信息
        mouth_curve = features["mouth_curvature"]["normalized_curvature"]
        cv2.putText(image, f"Mouth Curve: {mouth_curve:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 添加眼睛信息
        eye_closure = features["eye_closure"]["average"]
        cv2.putText(image, f"Eye Closure: {eye_closure:.2f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


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
        elif emotion=='surprised':self.colors['face']=(255,219,251)
        elif emotion=='neutral':self.colors['face']=(255,230,0)
        elif emotion=='disgusted':self.colors['face']=(191,255,163)
        elif emotion=='fearful':self.colors['face']=(223,163,155)
        
    
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
    emo = DeepFace.analyze(image_path, actions = ['emotion'],enforce_detection=False,)
    dom=emo[0]['dominant_emotion']
    print(emo)
    return dom

def main(image_path, output_path, predictor_path):
    """主函数，将人脸图像转换为 emoji
    
    Args:
        image_path: 输入图像路径
        output_path: 输出 emoji 路径
        predictor_path: dlib 面部特征预测器路径
    """
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化面部特征分析器
    analyzer = FacialFeatureAnalyzer(predictor_path)
    
    # 分析人脸
    features, message = analyzer.analyze_image(image_path)
    
    if features is None:
        print(f"错误: {message}")
        return False
    
    print(f"分析完成: {message}")
    print(f"嘴巴弧度: {features['mouth_curvature']['normalized_curvature']:.2f}")
    print(f"眼睛闭合度: {features['eye_closure']['average']:.2f}")
    
    emotion = emotion_recognition(image_path)
    # 初始化 emoji 生成器
    generator = EmojiGenerator(emotion=emotion)
    
    # 生成 emoji
    emoji = generator.generate_emoji(features)
    
    # 保存 emoji 图像
    output_file = generator.save_emoji(emoji, output_path)
    
    print(f"成功生成 emoji: {output_file}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将人脸图像转换为表情符号")
    parser.add_argument("image_path", help="输入人脸图像的路径")
    parser.add_argument("--output", default="output_emoji.png", help="输出emoji图像的路径")
    parser.add_argument(
        "--predictor", 
        default="shape_predictor_68_face_landmarks.dat",
        help="dlib面部特征点预测模型路径"
    )
    
    args = parser.parse_args()
    
    # 执行转换
    main(args.image_path, args.output, args.predictor)
