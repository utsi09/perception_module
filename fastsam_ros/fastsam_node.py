#!/usr/bin/env python3
"""
LiDAR + YOLO Segmentation Fusion Node

- LiDAR 포인트를 카메라로 투영
- YOLO 세그로 얻은 마스크 & 클래스 정보를 이용해
  각 객체까지의 LiDAR 기반 거리를 계산하고
  이미지에 표시해서 퍼블리시
"""

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

import sensor_msgs_py.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros

from ultralytics import YOLO


class LidarYoloFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_yolo_fusion')

        self.bridge = CvBridge()

        # ==========================
        # Parameters
        # ==========================
        self.declare_parameter('lidar_topic', '/carla/hero/lidar')
        self.declare_parameter('image_topic', '/carla/hero/rgb_front/image')
        self.declare_parameter('output_topic', '/fusion/overlay')

        # YOLO 관련 파라미터
        self.declare_parameter('model_path', 'yolo11n-seg.pt')
        self.declare_parameter('imgsz', 1280)
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('iou', 0.9)
        self.declare_parameter('device', 'cuda')

        # depth 표시 범위 (m)
        self.declare_parameter('max_depth', 60.0)

        # 값 읽기
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        self.model_path = self.get_parameter('model_path').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.device = self.get_parameter('device').value

        self.max_depth = float(self.get_parameter('max_depth').value)

        # ==========================
        # LiDAR ↔ Camera Extrinsic / Intrinsic
        # ==========================

        # LiDAR → Camera 변환 행렬 (예: CARLA 세팅)
        # Camera_X = -LiDAR_Y
        # Camera_Y = -LiDAR_Z
        # Camera_Z =  LiDAR_X
        # Camera 위치: LiDAR 기준 (2.0, 0.0, -0.4)
        self.T_lidar2cam = np.array([
            [0, -1,  0,  0.0],
            [0,  0, -1, -0.4],
            [1,  0,  0, -2.0],
            [0,  0,  0,  1.0]
        ], dtype=np.float32)

        # 카메라 내參 (예시: 1024x768, fx=fy=512, cx=512, cy=384)
        self.K = np.array([
            [512.0,   0.0, 512.0],
            [  0.0, 512.0, 384.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)

        # ==========================
        # Static TF (lidar → camera)
        # ==========================
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_tf()

        # ==========================
        # YOLO 모델 로드
        # ==========================
        self.get_logger().info(f'Loading YOLO segmentation model: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names  # id → name 매핑
        self.get_logger().info('YOLO model loaded')

        # ==========================
        # ROS 통신 설정
        # ==========================
        # message_filters 를 이용한 동기화
        lidar_sub = Subscriber(self, PointCloud2, self.lidar_topic)
        image_sub = Subscriber(self, Image, self.image_topic)

        self.ts = ApproximateTimeSynchronizer(
            [lidar_sub, image_sub],
            queue_size=10,
            slop=0.1  # 100ms 안쪽이면 같은 타임스탬프로 간주
        )
        self.ts.registerCallback(self.sync_callback)

        # 결과 이미지 퍼블리셔
        self.overlay_pub = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info('LidarYoloFusionNode initialized.')

    # -------------------------------------------------------------------------
    # Static TF publish (optional, rviz용)
    # -------------------------------------------------------------------------
    def publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'lidar'
        t.child_frame_id = 'rgb_front'

        # LiDAR 기준 카메라 위치 (LiDAR → 카메라)
        t.transform.translation.x = -2.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.4

        # 회전은 여기서는 단순 예시 (R 부분에 맞춰 조정 필요할 수도 있음)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info('Static TF (lidar → rgb_front) published.')

    # -------------------------------------------------------------------------
    # 메인 콜백: LiDAR + Image 동시 처리
    # -------------------------------------------------------------------------
    def sync_callback(self, lidar_msg: PointCloud2, image_msg: Image):
        """동기화된 LiDAR 포인트클라우드와 카메라 이미지를 동시에 처리"""

        # ==========================
        # 1) 이미지 로딩 + YOLO 세그멘테이션
        # ==========================
        try:
            img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        img_h, img_w = img.shape[:2]

        start_time = time.time()
        try:
            results = self.model(
                img,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                retina_masks=True,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f'YOLO inference error: {e}')
            return

        infer_ms = (time.time() - start_time) * 1000.0
        self.get_logger().info(f'YOLO inference time: {infer_ms:.2f} ms')

        r = results[0]
        result_img = r.plot()  # 상자/마스크 그려진 이미지 (기본 제공)

        # 세그 결과가 없으면 그냥 퍼블리시하고 끝
        if r.masks is None or r.boxes is None or r.boxes.cls is None:
            self.get_logger().info('No masks detected, publishing YOLO result only.')
            self.publish_image(result_img, image_msg.header)
            return

        # 마스크, 클래스, 박스 정보 꺼내기
        masks = r.masks.data.cpu().numpy()      # (N, Hm, Wm)
        classes = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        boxes = r.boxes.xyxy.cpu().numpy()      # (N, 4)

        num_obj = masks.shape[0]
        self.get_logger().info(f'Detected objects with masks: {num_obj}')

        # ==========================
        # 2) LiDAR 포인트 읽기
        # ==========================
        points = []
        for p in pc2.read_points(
            lidar_msg,
            skip_nans=True,
            field_names=("x", "y", "z", "intensity")
        ):
            if p is None or len(p) < 3:
                continue
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            points.append([x, y, z, 1.0])

        if len(points) == 0:
            self.get_logger().warn('No valid LiDAR points.')
            self.publish_image(result_img, image_msg.header)
            return

        pts_lidar = np.array(points, dtype=np.float32).T  # (4, N)

        # ==========================
        # 3) LiDAR → Camera 변환 + 투영
        # ==========================
        pts_cam = self.T_lidar2cam @ pts_lidar  # (4, N)

        # 카메라 앞쪽 (Z>0) 포인트만 사용
        z_cam = pts_cam[2, :]
        front_mask = z_cam > 0.0
        if not np.any(front_mask):
            self.get_logger().warn('No LiDAR points in front of camera.')
            self.publish_image(result_img, image_msg.header)
            return

        pts_cam = pts_cam[:, front_mask]   # (4, N_front)
        z_cam = pts_cam[2, :]

        # 투영
        proj = self.K @ pts_cam[:3, :]     # (3, N_front)
        u = (proj[0, :] / proj[2, :]).astype(np.int32)
        v = (proj[1, :] / proj[2, :]).astype(np.int32)

        # 이미지 범위 안 포인트만 선택
        in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        if not np.any(in_img):
            self.get_logger().warn('No LiDAR points projected into image.')
            self.publish_image(result_img, image_msg.header)
            return

        u = u[in_img]
        v = v[in_img]
        pts_cam = pts_cam[:, in_img]   # (4, N_valid)
        z_cam = pts_cam[2, :]

        # 이 좌표들(pts_cam[:3], u, v)을 기반으로 각 객체별로 필터링할 것

        # ==========================
        # 4) 각 객체별로 마스크에 속한 LiDAR 포인트만 골라서 거리 계산
        # ==========================
        for idx in range(num_obj):
            cls_id = int(classes[idx])
            cls_name = str(self.class_names.get(cls_id, cls_id))

            # YOLO 마스크를 원본 이미지 크기로 리사이즈
            mask = masks[idx]  # (Hm, Wm)
            mask_resized = cv2.resize(
                mask,
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )
            mask_bool = mask_resized > 0.5  # True/False

            # 현재 객체 마스크 안에 들어가는 LiDAR 포인트만 선택
            in_mask = mask_bool[v, u]   # (N_valid,) → True/False

            if not np.any(in_mask):
                # 이 객체에 해당하는 LiDAR 점이 없음
                continue

            pts_obj = pts_cam[:, in_mask]      # (4, N_obj)
            z_obj = pts_obj[2, :]
            dists = np.linalg.norm(pts_obj[:3, :], axis=0)  # 유클리드 거리

            # 대표 거리: 중앙값 사용 (outlier에 덜 민감)
            dist_med = float(np.median(dists))
            dist_min = float(np.min(dists))

            # 로그 출력
            self.get_logger().info(
                f'Object[{idx}] class={cls_name} '
                f'points={pts_obj.shape[1]} '
                f'dist_med={dist_med:.2f} m, dist_min={dist_min:.2f} m'
            )

            # ==========================
            # 5) 결과 이미지에 거리 텍스트 쓰기
            # ==========================
            x1, y1, x2, y2 = boxes[idx].astype(int)
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            text = f'{cls_name} {dist_min:.1f}m'


            font_scale = 1.2     
            thickness = 3        

            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            text_x = cx - text_w // 2
            text_y = cy + text_h // 2
            cv2.putText(
                result_img,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),   # 흰색
                thickness,
                cv2.LINE_AA
            )

        # ==========================
        # 6) 최종 이미지 퍼블리시
        # ==========================
        self.publish_image(result_img, image_msg.header)

    # -------------------------------------------------------------------------
    def publish_image(self, img_bgr, header):
        """OpenCV 이미지를 ROS Image 메세지로 변환해서 퍼블리시"""
        msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')
        msg.header = header  # 타임스탬프/프레임 그대로 사용
        self.overlay_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarYoloFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
