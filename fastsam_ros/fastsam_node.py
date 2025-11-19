#!/usr/bin/env python3
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

        self.declare_parameter('lidar_topic', '/carla/hero/lidar')
        self.declare_parameter('front_topic', '/carla/hero/rgb_front/image')
        self.declare_parameter('left_topic',  '/carla/hero/rgb_left/image')
        self.declare_parameter('right_topic', '/carla/hero/rgb_right/image')
        self.declare_parameter('back_topic',  '/carla/hero/rgb_back/image')

        # 공통 출력 토픽 이름 (지금은 안 씀)
        self.declare_parameter('output_topic_base', '/fusion')

        # YOLO 관련 파라미터
        self.declare_parameter('model_path', 'yolo11n-seg.pt')
        self.declare_parameter('imgsz', 1280)
        self.declare_parameter('conf', 0.6)
        self.declare_parameter('iou', 0.7)
        self.declare_parameter('device', 'cuda')

        # depth 표시 범위 (m) - 필요하면 필터에 사용 가능
        self.declare_parameter('max_depth', 60.0)

        # 값 읽기
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.front_topic = self.get_parameter('front_topic').value
        self.left_topic  = self.get_parameter('left_topic').value
        self.right_topic = self.get_parameter('right_topic').value
        self.back_topic  = self.get_parameter('back_topic').value

        self.output_topic_base = self.get_parameter('output_topic_base').value

        self.model_path = self.get_parameter('model_path').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.device = self.get_parameter('device').value
        self.max_depth = float(self.get_parameter('max_depth').value)


        # LiDAR ↔ Camera Extrinsic / Intrinsic


        # LiDAR → Camera 변환 행렬 (CARLA 기준, 이전에 쓰던 값)
        # LiDAR frame -> 각 카메라 frame
        self.T_lidar2cam_front = np.array([
            [0, -1,  0,  0.0],
            [0,  0, -1, -0.4],
            [1,  0,  0, -2.0],
            [0,  0,  0,  1.0]
        ], dtype=np.float32)

        self.T_lidar2cam_left = np.array([
            [ 1,  0,  0,   1.0],
            [ 0,  0, -1,  -1.4],
            [ 0,  1,  0,   0.0],
            [ 0,  0,  0,   1.0]
        ], dtype=np.float32)

        self.T_lidar2cam_right = np.array([
            [ -1,  0,  0,  -1.0],
            [  0,  0, -1,  -1.4],
            [  0, -1,  0,   0.0],
            [  0,  0,  0,   1.0]
        ], dtype=np.float32)

        self.T_lidar2cam_back = np.array([
            [ 0,  1,  0,  0.0],
            [ 0,  0, -1, -0.4],
            [-1,  0,  0,  2.0],
            [ 0,  0,  0,  1.0]
        ], dtype=np.float32)

        # 순서: front, left, right, back
        self.extrinsic_list = [
            self.T_lidar2cam_front,
            self.T_lidar2cam_left,
            self.T_lidar2cam_right,
            self.T_lidar2cam_back
        ]

        # LiDAR -> Camera의 역행렬 (Camera -> LiDAR) 미리 계산 (TopView용)
        self.cam2lidar_list = [np.linalg.inv(T) for T in self.extrinsic_list]

        # 카메라 intrinsic (CARLA RGB 1024x768, f ~ 512로 가정)
        self.K = np.array([
            [512.0,   0.0, 512.0],
            [  0.0, 512.0, 384.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)


        # Static TF (예: front 기준 lidar 위치)

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_tf()


        # YOLO 모델 로드

        self.get_logger().info(f'Loading YOLO segmentation model: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        self.get_logger().info('YOLO model loaded')


        # ROS 통신 설정

        lidar_sub  = Subscriber(self, PointCloud2, self.lidar_topic)
        front_sub  = Subscriber(self, Image, self.front_topic)
        left_sub   = Subscriber(self, Image, self.left_topic)
        right_sub  = Subscriber(self, Image, self.right_topic)
        back_sub   = Subscriber(self, Image, self.back_topic)

        self.ts = ApproximateTimeSynchronizer(
            [lidar_sub, front_sub, left_sub, right_sub, back_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)


        # 카메라별 overlay 이미지 토픽
        self.overlay_pub_front = self.create_publisher(
            Image, f'{self.output_topic_base}/front/overlay', 10)
        self.overlay_pub_left = self.create_publisher(
            Image, f'{self.output_topic_base}/left/overlay', 10)
        self.overlay_pub_right = self.create_publisher(
            Image, f'{self.output_topic_base}/right/overlay', 10)
        self.overlay_pub_back = self.create_publisher(
            Image, f'{self.output_topic_base}/back/overlay', 10)

        # 매핑용 딕셔너리
        self.overlay_pubs = {
            'front': self.overlay_pub_front,
            'left':  self.overlay_pub_left,
            'right': self.overlay_pub_right,
            'back':  self.overlay_pub_back,
        }

        # TopView 토픽
        self.topview_pub = self.create_publisher(
            Image, f'{self.output_topic_base}/topview', 10)

        # TopView 스무딩용 center 저장
        self.prev_centers = {}

        self.get_logger().info('LidarYoloFusionNode initialized.')


    def publish_static_tf(self):

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'rgb_front'
        t.child_frame_id = 'lidar'

        t.transform.translation.x = -2.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.4

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)


    def sync_callback(self,
                      lidar_msg: PointCloud2,
                      front_msg: Image,
                      left_msg: Image,
                      right_msg: Image,
                      back_msg: Image):


        # LiDAR 포인트 한 번만 읽기
        pts_list = []
        for p in pc2.read_points(
            lidar_msg,
            skip_nans=True,
            field_names=("x", "y", "z", "intensity")
        ):
            # [x, y, z, 1] (homogeneous)
            pts_list.append([p[0], p[1], p[2], 1.0])

        if len(pts_list) == 0:
            self.get_logger().warn('No LiDAR points.')
            return

        pts_lidar = np.array(pts_list, dtype=np.float32).T   # (4, N)

        # 카메라별 이미지 메시지 / 이름 / 행렬 묶기
        image_msgs = [front_msg, left_msg, right_msg, back_msg]
        cam_names  = ['front', 'left', 'right', 'back']

        # TopView에 쓸 전체 객체 리스트 (LiDAR frame 기준)
        all_centers_lidar = []
        all_labels = []
        all_dists = []

        # 카메라마다 별도로 처리
        for cam_idx, (image_msg, cam_name) in enumerate(zip(image_msgs, cam_names)):
            try:
                img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception:
                continue

            img_h, img_w = img.shape[:2]

            #  LiDAR → 현재 카메라 좌표계
            T_lidar2cam = self.extrinsic_list[cam_idx]
            pts_cam = T_lidar2cam @ pts_lidar  # (4, N)

            # 카메라 앞에 있는 점만 사용 (z > 0)
            z_cam = pts_cam[2, :]
            front_mask = z_cam > 0
            if not np.any(front_mask):
                # 이 카메라 쪽으로 보이는 LiDAR 점이 없다
                self.publish_image(img, image_msg.header, cam_name)
                continue

            pts_cam = pts_cam[:, front_mask]
            z_cam = z_cam[front_mask]

            #  카메라 intrinsics로 픽셀 좌표로 프로젝션
            proj = self.K @ pts_cam[:3, :]    # (3, N)
            u = (proj[0, :] / proj[2, :]).astype(np.int32)
            v = (proj[1, :] / proj[2, :]).astype(np.int32)

            in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            if not np.any(in_img):
                self.publish_image(img, image_msg.header, cam_name)
                continue

            pts_cam = pts_cam[:, in_img]
            u = u[in_img]
            v = v[in_img]

            # YOLO 세그멘테이션 실행
            start_time = time.time()
            results = self.model(
                img,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                retina_masks=True,
                verbose=False
            )
            infer_ms = (time.time() - start_time) * 1000.0
            self.get_logger().info(
                f'[{cam_name}] YOLO inference time: {infer_ms:.2f} ms')

            r = results[0]
            result_img = r.plot()

            if r.masks is None:
                # 객체가 없으면 그냥 YOLO 결과만 출력
                self.publish_image(result_img, image_msg.header, cam_name)
                continue

            masks = r.masks.data.cpu().numpy()  # (num_obj, Hm, Wm)
            classes = r.boxes.cls.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy()
            num_obj = masks.shape[0]

            cam2lidar = self.cam2lidar_list[cam_idx]

            # 이 카메라에서 검출된 객체들의 center/distance를 TopView용으로 저장
            for obj_idx in range(num_obj):
                cls_id = classes[obj_idx]
                cls_name = self.class_names.get(cls_id, str(cls_id))

                mask = masks[obj_idx]
                # 마스크 해상도를 현재 이미지 크기로 리사이즈
                mask_r = cv2.resize(
                    mask,
                    (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST
                )
                mask_bool = mask_r > 0.5

                # 프로젝션된 LiDAR 포인트 중, 이 객체 마스크 안에 들어가는 것만 선택
                in_mask = mask_bool[v, u]
                if not np.any(in_mask):
                    continue

                pts_obj_cam = pts_cam[:, in_mask]  # (4, M)
                dists = np.linalg.norm(pts_obj_cam[:3, :], axis=0)
                dist_min = float(np.min(dists))

                # 중심점 (카메라 좌표계에서 평균)
                center_cam = np.mean(pts_obj_cam[:3, :], axis=1)  # (3,)

                # 중심점을 LiDAR(차량) 좌표계로 변환 (TopView용)
                center_cam_h = np.concatenate([center_cam, [1.0]])  # (4,)
                center_lidar_h = cam2lidar @ center_cam_h
                center_lidar = center_lidar_h[:3]

                all_centers_lidar.append(center_lidar)
                all_labels.append(cls_name)
                all_dists.append(dist_min)

                # 박스 중앙에 거리 텍스트 표시
                x1, y1, x2, y2 = boxes[obj_idx].astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                text = f"{cls_name} {dist_min:.1f}m"
                fs = 0.7
                th = 2
                cv2.putText(
                    result_img, text, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th
                )

            # 이 카메라의 overlay 이미지 퍼블리시
            self.publish_image(result_img, image_msg.header, cam_name)

        # 3) TopView 이미지 생성
        if len(all_centers_lidar) > 0:
            topview = self.build_topview(all_centers_lidar, all_labels, all_dists)
            # 시간은 일단 front 카메라 헤더 사용
            self.publish_topview(topview, front_msg.header)

    # -------------------------------------------------------------------------
    def build_topview(self, centers_lidar, labels, dists):
        W, H = 800, 800
        img = np.ones((H, W, 3), dtype=np.uint8) * 255

        # 스케일 (픽셀/미터)
        scale = 12.0   # 1m -> 5px (필요시 조절)
        origin_x = W // 2
        origin_y = H//2   # ego 차량을 아래쪽에 두고 위로 갈수록 전방

        # ego 차량 그리기
        car_w, car_h = 40, 70
        cv2.rectangle(
            img,
            (origin_x - car_w // 2, origin_y - car_h),
            (origin_x + car_w // 2, origin_y),
            (180, 180, 180),
            -1
        )
        cv2.putText(
            img, "ego",
            (origin_x - 20, origin_y - car_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

        # 객체 그리기
        for center_lidar, label, dist in zip(centers_lidar, labels, dists):
            l = label.lower()
            if ("car" not in l and
                "vehicle" not in l and
                "truck" not in l and
                "person" not in l):
                # 관심 객체만 그림
                continue

            X, Y, Z = center_lidar  # LiDAR frame (X front, Y right, Z up)

            # TopView 좌표로 변환
            px = int(origin_x - Y * scale)     # 오른쪽이 +Y
            py = int(origin_y - X * scale)     # 위쪽이 +X

            # 스무딩 키 (라벨 + 위치 대략 이용)
            key = f"{label}_{int(X)}_{int(Y)}"
            alpha = 0.7

            if key not in self.prev_centers:
                self.prev_centers[key] = np.array([px, py], dtype=float)
            else:
                self.prev_centers[key] = (
                    alpha * self.prev_centers[key] +
                    (1.0 - alpha) * np.array([px, py], dtype=float)
                )

            px_s, py_s = self.prev_centers[key].astype(int)

            # ego와 연결선
            cv2.line(
                img,
                (origin_x, origin_y - car_h // 2),
                (px_s, py_s),
                (0, 0, 0),
                2
            )

            # 거리 텍스트
            mid_x = (origin_x + px_s) // 2
            mid_y = (origin_y - car_h // 2 + py_s) // 2
            cv2.putText(
                img,
                f"{dist:.1f}m",
                (mid_x - 20, mid_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            # 객체 모양
            if "car" in l or "vehicle" in l or "truck" in l:
                cv2.rectangle(
                    img,
                    (px_s - 15, py_s - 25),
                    (px_s + 15, py_s + 25),
                    (255, 0, 0),
                    -1
                )
            elif "person" in l:
                cv2.circle(img, (px_s, py_s), 18, (0, 200, 0), -1)

        return img


    def publish_image(self, img, header, cam_name):

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header = header

        pub = self.overlay_pubs.get(cam_name, None)
        if pub is not None:
            pub.publish(msg)


    def publish_topview(self, img, header):
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header = header
        self.topview_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = LidarYoloFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
