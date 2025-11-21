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
        self.declare_parameter('output_topic_base', '/fusion')
        self.declare_parameter('model_path', 'yolo11n-seg.pt')
        self.declare_parameter('imgsz', 1280)
        self.declare_parameter('conf', 0.6)
        self.declare_parameter('iou', 0.7)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('max_depth', 60.0)

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

        self.extrinsic_list = [
            self.T_lidar2cam_front,
            self.T_lidar2cam_left,
            self.T_lidar2cam_right,
            self.T_lidar2cam_back
        ]

        self.cam2lidar_list = [np.linalg.inv(T) for T in self.extrinsic_list]

        self.K = np.array([
            [512.0,   0.0, 512.0],
            [  0.0, 512.0, 384.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float32)

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_tf()

        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

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

        self.overlay_pub_front = self.create_publisher(Image, f'{self.output_topic_base}/front/overlay', 10)
        self.overlay_pub_left  = self.create_publisher(Image, f'{self.output_topic_base}/left/overlay', 10)
        self.overlay_pub_right = self.create_publisher(Image, f'{self.output_topic_base}/right/overlay', 10)
        self.overlay_pub_back  = self.create_publisher(Image, f'{self.output_topic_base}/back/overlay', 10)

        self.overlay_pubs = {
            'front': self.overlay_pub_front,
            'left':  self.overlay_pub_left,
            'right': self.overlay_pub_right,
            'back':  self.overlay_pub_back
        }

        self.topview_pub = self.create_publisher(Image, f'{self.output_topic_base}/topview', 10)
        self.infer_cache = []
        self.prev_centers = {}

    def publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'rgb_front'
        t.child_frame_id = 'lidar'
        t.transform.translation.x = -2.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.4
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def sync_callback(self, lidar_msg, front_msg, left_msg, right_msg, back_msg):
        st_t = time.time()
        pts_list = []
        for p in pc2.read_points(lidar_msg, skip_nans=True, field_names=("x","y","z","intensity")):
            pts_list.append([p[0], p[1], p[2], 1.0])

        if len(pts_list) == 0:
            return

        pts_lidar = np.array(pts_list, dtype=np.float32).T

        image_msgs = [front_msg, left_msg, right_msg, back_msg]
        cv_msgs = [self.bridge.imgmsg_to_cv2(m, 'bgr8') for m in image_msgs]
        cam_names = ['front','left','right','back']

        all_centers_lidar = []
        all_labels = []
        all_dists = []
        pts_cams = [None]*4
        uvs = [None]*4

        for cam_idx, (image_msg, cam_name) in enumerate(zip(image_msgs, cam_names)):
            img = cv_msgs[cam_idx]
            img_h, img_w = img.shape[:2]

            T_lidar2cam = self.extrinsic_list[cam_idx]
            pts_cam = T_lidar2cam @ pts_lidar

            z_cam = pts_cam[2,:]
            front_mask = z_cam > 0
            if not np.any(front_mask):
                continue

            pts_cam = pts_cam[:, front_mask]
            proj = self.K @ pts_cam[:3,:]
            u = (proj[0]/proj[2]).astype(np.int32)
            v = (proj[1]/proj[2]).astype(np.int32)

            in_img = (u>=0)&(u<img_w)&(v>=0)&(v<img_h)
            if not np.any(in_img):
                continue

            pts_cams[cam_idx] = pts_cam[:, in_img]
            uvs[cam_idx] = (u[in_img], v[in_img])

        start = time.time()
        results = self.model(
            cv_msgs,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            retina_masks=True,
            verbose=False
        )

        for i, cam_name in enumerate(cam_names):
            r_list = results[i]

            if len(r_list) == 0:
                # YOLO 결과가 아예 없는 경우
                self.publish_image(cv_msgs[i], image_msgs[i].header, cam_names[i])
                continue

            r = r_list[0]
            result_img = r.plot()
            img_h, img_w = result_img.shape[:2]
            orig_msg = image_msgs[i]

            if r.masks is None:
                self.publish_image(result_img, orig_msg.header, cam_name)
                continue

            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy()
            num_obj = masks.shape[0]
            cam2lidar = self.cam2lidar_list[i]

            if uvs[i] is None or pts_cams[i] is None:
                self.publish_image(result_img, orig_msg.header, cam_name)
                continue

            u, v = uvs[i]

            for obj_idx in range(num_obj):
                cls_id = classes[obj_idx]
                cls_name = self.class_names.get(cls_id, str(cls_id))
                mask = masks[obj_idx]

                mask_r = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_r > 0.5

                in_mask = mask_bool[v, u]
                if not np.any(in_mask):
                    continue

                pts_obj_cam = pts_cams[i][:, in_mask]
                dists = np.linalg.norm(pts_obj_cam[:3,:], axis=0)
                dist_min = float(np.min(dists))

                center_cam = np.mean(pts_obj_cam[:3,:], axis=1)
                center_cam_h = np.concatenate([center_cam,[1.0]])
                center_lidar_h = cam2lidar @ center_cam_h
                center_lidar = center_lidar_h[:3]

                all_centers_lidar.append(center_lidar)
                all_labels.append(cls_name)
                all_dists.append(dist_min)

                x1,y1,x2,y2 = boxes[obj_idx].astype(int)
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)
                cv2.putText(result_img, f"{cls_name} {dist_min:.1f}m",
                            (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2)

            self.publish_image(result_img, orig_msg.header, cam_name)

        if len(all_centers_lidar) > 0:
            top = self.build_topview(all_centers_lidar, all_labels, all_dists)
            self.publish_topview(top, front_msg.header)
        
        infer_time = (time.time() - st_t) * 1000
        self.infer_cache.append(infer_time)
        if len(self.infer_cache) > 100:
            self.get_logger().info(f' 100개 평균 추론 시간은 {np.mean(self.infer_cache)} ms')
            self.infer_cache.clear()    
        self.get_logger().info(f'Inference Time : {infer_time} ms')

    def build_topview(self, centers_lidar, labels, dists):
        W,H = 800,800
        img = np.ones((H,W,3), np.uint8)*255
        scale = 12.0
        origin_x = W//2
        origin_y = H//2

        car_w,car_h = 40,70
        cv2.rectangle(img, (origin_x-car_w//2, origin_y-car_h),
                           (origin_x+car_w//2, origin_y),
                           (180,180,180), -1)

        cv2.putText(img, "ego", (origin_x-20, origin_y-car_h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)

        for center_lidar, label, dist in zip(centers_lidar, labels, dists):
            l = label.lower()
            if ("car" not in l and "vehicle" not in l and
                "truck" not in l and "person" not in l):
                continue

            X,Y,Z = center_lidar
            px = int(origin_x - Y*scale)
            py = int(origin_y - X*scale)

            cv2.line(img, (origin_x, origin_y-car_h//2),
                          (px,py), (0,0,0),2)

            mid_x = (origin_x+px)//2
            mid_y = (origin_y-car_h//2 + py)//2
            cv2.putText(img, f"{dist:.1f}m", (mid_x-20, mid_y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)

            if "car" in l or "vehicle" in l or "truck" in l:
                cv2.rectangle(img, (px-15,py-25),(px+15,py+25),(255,0,0),-1)
            elif "person" in l:
                cv2.circle(img, (px,py), 18,(0,200,0),-1)

        return img

    def publish_image(self, img, header, cam_name):
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header = header
        pub = self.overlay_pubs.get(cam_name)
        if pub:
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
