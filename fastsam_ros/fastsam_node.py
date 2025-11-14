#!/usr/bin/env python3
"""
LiDAR + YOLO Segmentation Fusion Node (+TOPVIEW)

- LiDAR 포인트를 카메라로 투영
- YOLO 세그로 얻은 마스크 & 클래스 정보를 이용해
  각 객체까지의 LiDAR 기반 거리를 계산하고
  이미지에 표시해서 퍼블리시
- 추가: Top-View 시각화 이미지 생성 후 ROS 토픽으로 발행
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
        self.T_lidar2cam = np.array([
            [0, -1,  0,  0.0],
            [0,  0, -1, -0.4],
            [1,  0,  0, -2.0],
            [0,  0,  0,  1.0]
        ], dtype=np.float32)

        # 카메라 intrinsic
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
        self.class_names = self.model.names
        self.get_logger().info('YOLO model loaded')

        # ==========================
        # ROS 통신 설정
        # ==========================
        lidar_sub = Subscriber(self, PointCloud2, self.lidar_topic)
        image_sub = Subscriber(self, Image, self.image_topic)

        self.ts = ApproximateTimeSynchronizer(
            [lidar_sub, image_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # 결과 이미지 퍼블리셔
        self.overlay_pub = self.create_publisher(Image, self.output_topic, 10)

        # TOPVIEW 퍼블리셔 추가
        self.topview_pub = self.create_publisher(Image, "/fusion/topview", 10)

        self.get_logger().info('LidarYoloFusionNode initialized.')

    # -------------------------------------------------------------------------
    def publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'lidar'
        t.child_frame_id = 'rgb_front'

        t.transform.translation.x = -2.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.4

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    # -------------------------------------------------------------------------
    def sync_callback(self, lidar_msg: PointCloud2, image_msg: Image):
        """LiDAR + RGB 동기화 처리"""

        try:
            img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except:
            return

        img_h, img_w = img.shape[:2]

        # YOLO 수행
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
        self.get_logger().info(f'YOLO inference time: {infer_ms:.2f} ms')

        r = results[0]
        result_img = r.plot()

        if r.masks is None:
            self.publish_image(result_img, image_msg.header)
            return

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy()

        num_obj = masks.shape[0]

        # LiDAR 포인트 읽기
        pts_list = []
        for p in pc2.read_points(lidar_msg, skip_nans=True,
                                 field_names=("x","y","z","intensity")):
            pts_list.append([p[0],p[1],p[2],1.0])

        if len(pts_list)==0:
            self.publish_image(result_img, image_msg.header)
            return

        pts_lidar = np.array(pts_list,dtype=np.float32).T

        # LiDAR → Camera
        pts_cam = self.T_lidar2cam @ pts_lidar
        z_cam = pts_cam[2,:]
        front_mask = z_cam > 0
        pts_cam = pts_cam[:,front_mask]
        z_cam = z_cam[front_mask]

        proj = self.K @ pts_cam[:3,:]
        u = (proj[0,:] / proj[2,:]).astype(np.int32)
        v = (proj[1,:] / proj[2,:]).astype(np.int32)

        in_img = (u>=0)&(u<img_w)&(v>=0)&(v<img_h)
        pts_cam = pts_cam[:,in_img]
        u = u[in_img]
        v = v[in_img]

        obj_centers=[]
        obj_labels=[]
        obj_dists=[]

        # 객체별 처리
        for idx in range(num_obj):
            cls_id = classes[idx]
            cls_name = self.class_names.get(cls_id,str(cls_id))

            mask = masks[idx]
            mask_r = cv2.resize(mask,(img_w,img_h),
                                interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_r > 0.5

            in_mask = mask_bool[v,u]
            if not np.any(in_mask):
                continue

            pts_obj = pts_cam[:,in_mask]
            dists = np.linalg.norm(pts_obj[:3,:],axis=0)
            dist_min=float(np.min(dists))

            # 중심 계산
            center = np.mean(pts_obj[:3,:],axis=1)
            obj_centers.append(center)
            obj_labels.append(cls_name)
            obj_dists.append(dist_min)

            # 박스 중앙에 거리 표시
            x1,y1,x2,y2 = boxes[idx].astype(int)
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)

            text=f"{cls_name} {dist_min:.1f}m"
            fs=1.2; th=3
            (tw,th2),base=cv2.getTextSize(
                text,cv2.FONT_HERSHEY_SIMPLEX,fs,th)
            tx = cx - tw//2
            ty = cy + th2//2
            cv2.putText(result_img,text,(tx,ty),
                        cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),th)

        # overlay 발행
        self.publish_image(result_img, image_msg.header)

        # --------------------------
        # TOP VIEW 생성 + 발행
        # --------------------------
        topview = self.build_topview(obj_centers,obj_labels,obj_dists)
        self.publish_topview(topview,image_msg.header)

    # -------------------------------------------------------------------------
    def build_topview(self, centers, labels, dists):
        W,H=600,800
        img=np.ones((H,W,3),dtype=np.uint8)*255

        scale=20.0
        origin_x=W//2
        origin_y=H-50

        # ego 차량
        car_w,car_h=40,70
        cv2.rectangle(img,(origin_x-car_w//2,origin_y-car_h),
                      (origin_x+car_w//2,origin_y),
                      (180,180,180),-1)
        cv2.putText(img,"ego",(origin_x-20,origin_y-car_h-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

        for (center,label,dist) in zip(centers,labels,dists):
            X,Y,Z=center
            px=int(origin_x+X*scale)
            py=int(origin_y-Z*scale)

            # 연결선
            cv2.line(img,(origin_x,origin_y-car_h//2),(px,py),(0,0,0),2)

            # 거리 텍스트
            mid_x=(origin_x+px)//2
            mid_y=(origin_y-car_h//2+py)//2
            cv2.putText(img,f"{dist:.1f}m",(mid_x-20,mid_y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

            # 객체 형태
            l=label.lower()
            if "car" in l or "vehicle" in l or "truck" in l :
                cv2.rectangle(img,(px-15,py-25),(px+15,py+25),(255,0,0),-1)
            elif "person" in l:
                cv2.circle(img,(px,py),18,(0,200,0),-1)
            else:
                tri=np.array([
                    [px,py-25],
                    [px-20,py+20],
                    [px+20,py+20]
                ],np.int32)
                cv2.fillPoly(img,[tri],(0,0,255))

        return img

    # -------------------------------------------------------------------------
    def publish_image(self,img,header):
        msg=self.bridge.cv2_to_imgmsg(img,encoding='bgr8')
        msg.header=header
        self.overlay_pub.publish(msg)

    # -------------------------------------------------------------------------
    def publish_topview(self,img,header):
        msg=self.bridge.cv2_to_imgmsg(img,encoding='bgr8')
        msg.header=header
        self.topview_pub.publish(msg)


# -------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node=LidarYoloFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
