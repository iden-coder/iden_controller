#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a stable, temporary cone cloud for the large-room global costmap."""

import json
import math
import threading
import time

import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Bool, Header, String
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion


def distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


class StableConeMapper(object):
    def __init__(self):
        rospy.init_node("xunfei2026_stable_cone_mapper")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.control_topic = rospy.get_param(
            "~control_topic", "/factory_room/navigation_active")
        self.cloud_topic = rospy.get_param(
            "~cloud_topic", "/factory_room/stable_cone_cloud")
        self.status_topic = rospy.get_param(
            "~status_topic", "/factory_room/cone_mapper_status")
        self.map_frame = rospy.get_param("~map_frame", "map")

        self.room_min_x = float(rospy.get_param("~room_min_x", -2.23))
        self.room_max_x = float(rospy.get_param("~room_max_x", 2.80))
        self.room_min_y = float(rospy.get_param("~room_min_y", -3.28))
        self.room_max_y = float(rospy.get_param("~room_max_y", -1.18))
        self.min_scan_range = float(rospy.get_param("~min_scan_range_m", 0.14))
        self.max_scan_range = float(rospy.get_param("~max_scan_range_m", 3.4))
        self.static_reject = float(rospy.get_param(
            "~static_wall_reject_distance_m", 0.11))
        self.cluster_gap = float(rospy.get_param("~cluster_gap_m", 0.10))
        self.cluster_min_points = int(rospy.get_param("~cluster_min_points", 2))
        self.cluster_max_points = int(rospy.get_param("~cluster_max_points", 24))
        self.cluster_max_diameter = float(rospy.get_param(
            "~cluster_max_diameter_m", 0.28))
        self.track_match = float(rospy.get_param("~track_match_m", 0.20))
        self.track_deduplicate = float(rospy.get_param(
            "~track_deduplicate_m", 0.22))
        self.confirm_scans = int(rospy.get_param("~confirm_scans", 3))
        self.track_ttl = float(rospy.get_param("~track_ttl_s", 5.0))
        self.track_alpha = float(rospy.get_param("~track_alpha", 0.35))
        self.cone_radius = float(rospy.get_param("~cone_radius_m", 0.10))
        self.max_published_cones = int(rospy.get_param(
            "~max_published_cones", 18))
        self.clear_cooldown = float(rospy.get_param(
            "~clear_costmaps_cooldown_s", 2.0))

        self.lock = threading.RLock()
        self.active = False
        self.map_info = None
        self.static_distance = None
        self.tracks = []
        self.next_track_id = 1
        self.robot_position = None
        self.last_clear_request = 0.0
        self.clear_in_progress = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cloud_pub = rospy.Publisher(
            self.cloud_topic, PointCloud2, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=5, latch=True)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback,
                         queue_size=1)
        rospy.Subscriber(self.control_topic, Bool, self.control_callback,
                         queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback,
                         queue_size=1, buff_size=2 ** 20)
        self.publish_timer = rospy.Timer(rospy.Duration(0.20), self.publish_timer_cb)
        rospy.on_shutdown(self.shutdown)
        self.publish_status("READY", active=False)

    def publish_status(self, state, **values):
        payload = {"state": state, "stamp": time.time()}
        payload.update(values)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.logwarn("STABLE_CONE_MAPPER %s", json.dumps(payload, ensure_ascii=False))

    def map_callback(self, msg):
        try:
            grid = np.asarray(msg.data, dtype=np.int16).reshape(
                (msg.info.height, msg.info.width))
            occupied = grid >= 65
            free_mask = np.logical_not(occupied).astype(np.uint8)
            distance_cells = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
            q = msg.info.origin.orientation
            origin_yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
            info = {
                "resolution": float(msg.info.resolution),
                "width": int(msg.info.width),
                "height": int(msg.info.height),
                "origin_x": float(msg.info.origin.position.x),
                "origin_y": float(msg.info.origin.position.y),
                "origin_yaw": origin_yaw,
            }
            with self.lock:
                self.map_info = info
                self.static_distance = distance_cells * info["resolution"]
            rospy.logwarn(
                "STABLE_CONE_STATIC_MAP_READY size=%dx%d resolution=%.3f",
                info["width"], info["height"], info["resolution"])
        except Exception as exc:
            rospy.logerr("stable cone map conversion failed: %s", exc)

    def control_callback(self, msg):
        requested = bool(msg.data)
        clear_needed = False
        with self.lock:
            if requested == self.active:
                return
            self.active = requested
            if not requested:
                clear_needed = bool(self.tracks)
                self.tracks = []
        self.publish_status("ACTIVE" if requested else "INACTIVE",
                            active=requested)
        if clear_needed:
            self.request_costmap_clear("mapper disabled")

    def world_to_cell(self, x, y):
        with self.lock:
            info = None if self.map_info is None else dict(self.map_info)
        if info is None:
            return None
        dx = x - info["origin_x"]
        dy = y - info["origin_y"]
        cosine = math.cos(info["origin_yaw"])
        sine = math.sin(info["origin_yaw"])
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        mx = int(math.floor(local_x / info["resolution"]))
        my = int(math.floor(local_y / info["resolution"]))
        if mx < 0 or my < 0 or mx >= info["width"] or my >= info["height"]:
            return None
        return mx, my

    def static_clearance(self, x, y):
        cell = self.world_to_cell(x, y)
        if cell is None:
            return 0.0
        with self.lock:
            if self.static_distance is None:
                return 0.0
            return float(self.static_distance[cell[1], cell[0]])

    def inside_room(self, x, y):
        return (self.room_min_x <= x <= self.room_max_x and
                self.room_min_y <= y <= self.room_max_y)

    def scan_callback(self, msg):
        with self.lock:
            if not self.active or self.static_distance is None:
                return
        source_frame = msg.header.frame_id or "laser_frame"
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, source_frame, rospy.Time(0), rospy.Duration(0.08))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "stable cone TF unavailable: %s", exc)
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = euler_from_quaternion(
            (rotation.x, rotation.y, rotation.z, rotation.w))[2]
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        with self.lock:
            self.robot_position = (translation.x, translation.y)
        clusters = []
        current = []
        previous = None

        def finish_cluster():
            if current:
                clusters.append(list(current))
                current[:] = []

        angle = msg.angle_min
        for value in msg.ranges:
            valid = (math.isfinite(value) and
                     max(self.min_scan_range, msg.range_min) <= value <=
                     min(self.max_scan_range, msg.range_max))
            if valid:
                laser_x = value * math.cos(angle)
                laser_y = value * math.sin(angle)
                world_x = translation.x + cosine * laser_x - sine * laser_y
                world_y = translation.y + sine * laser_x + cosine * laser_y
                point = (world_x, world_y)
                valid = (self.inside_room(world_x, world_y) and
                         self.static_clearance(world_x, world_y) >=
                         self.static_reject)
            if not valid:
                finish_cluster()
                previous = None
            else:
                if previous is not None and distance(previous, point) > self.cluster_gap:
                    finish_cluster()
                current.append(point)
                previous = point
            angle += msg.angle_increment
        finish_cluster()

        candidates = []
        for cluster in clusters:
            if not self.cluster_min_points <= len(cluster) <= self.cluster_max_points:
                continue
            diameter = distance(cluster[0], cluster[-1])
            if diameter > self.cluster_max_diameter:
                continue
            xs = sorted(point[0] for point in cluster)
            ys = sorted(point[1] for point in cluster)
            center = (xs[len(xs) // 2], ys[len(ys) // 2])
            if self.static_clearance(center[0], center[1]) < self.static_reject:
                continue
            candidates.append(center)
        self.update_tracks(candidates)

    def update_tracks(self, candidates):
        now = time.monotonic()
        removed_stable = False
        with self.lock:
            used = set()
            for candidate in candidates:
                best_index = None
                best_distance = self.track_match
                for index, track in enumerate(self.tracks):
                    if index in used:
                        continue
                    value = distance(candidate, (track["x"], track["y"]))
                    if value < best_distance:
                        best_distance = value
                        best_index = index
                if best_index is None:
                    self.tracks.append({
                        "id": self.next_track_id,
                        "x": candidate[0], "y": candidate[1],
                        "hits": 1, "last_seen": now, "stable": False,
                    })
                    self.next_track_id += 1
                    used.add(len(self.tracks) - 1)
                else:
                    track = self.tracks[best_index]
                    alpha = self.track_alpha
                    track["x"] = (1.0 - alpha) * track["x"] + alpha * candidate[0]
                    track["y"] = (1.0 - alpha) * track["y"] + alpha * candidate[1]
                    track["hits"] += 1
                    track["last_seen"] = now
                    if track["hits"] >= self.confirm_scans:
                        track["stable"] = True
                    used.add(best_index)

            kept = []
            for track in self.tracks:
                if now - track["last_seen"] <= self.track_ttl:
                    kept.append(track)
                elif track["stable"]:
                    removed_stable = True
            self.tracks = kept
            # Merge historical duplicates caused by localization jitter.  A
            # physical cone must produce one stable map identity, not a trail
            # of old centers as the robot moves around it.
            merged = []
            for track in sorted(self.tracks,
                                key=lambda value: value["last_seen"],
                                reverse=True):
                duplicate = next((item for item in merged if distance(
                    (track["x"], track["y"]),
                    (item["x"], item["y"])) < self.track_deduplicate), None)
                if duplicate is None:
                    merged.append(track)
                else:
                    old_hits = duplicate["hits"]
                    total_hits = max(1, old_hits + track["hits"])
                    duplicate["x"] = (
                        duplicate["x"] * old_hits +
                        track["x"] * track["hits"]) / total_hits
                    duplicate["y"] = (
                        duplicate["y"] * old_hits +
                        track["y"] * track["hits"]) / total_hits
                    duplicate["hits"] = total_hits
                    duplicate["stable"] = (
                        duplicate["stable"] or track["stable"])
                    duplicate["last_seen"] = max(
                        duplicate["last_seen"], track["last_seen"])
            self.tracks = merged
            stable_count = sum(1 for track in self.tracks if track["stable"])
        if removed_stable:
            # Cones are static during one run.  Rebuilding a 640x640 global
            # costmap every time an observation leaves the lidar view blocked
            # the controller for several seconds.  Keep the existing mark; the
            # whole room move_base is discarded at parking handoff.
            rospy.loginfo_throttle(
                2.0, "STABLE_CONE_TRACK_EXPIRED retained_in_costmap=true")
        rospy.loginfo_throttle(
            1.0, "STABLE_CONE_TRACKS candidates=%d tracks=%d stable=%d",
            len(candidates), len(self.tracks), stable_count)

    def cloud_points(self):
        with self.lock:
            tracks = [dict(track) for track in self.tracks if track["stable"]]
            robot = self.robot_position
        if robot is not None:
            tracks.sort(key=lambda track: distance(
                robot, (track["x"], track["y"])))
        tracks = tracks[:max(1, self.max_published_cones)]
        # A single center point is sufficient because costmap inflation owns
        # the exclusion radius.  Filled disks multiplied each cone into 21 TEB
        # obstacles and caused multi-second control-loop stalls.
        points = [(track["x"], track["y"], 0.05) for track in tracks]
        return points, tracks

    def publish_timer_cb(self, _event):
        with self.lock:
            active = self.active
        header = Header(stamp=rospy.Time.now(), frame_id=self.map_frame)
        if not active:
            self.cloud_pub.publish(pc2.create_cloud_xyz32(header, []))
            return
        points, tracks = self.cloud_points()
        self.cloud_pub.publish(pc2.create_cloud_xyz32(header, points))
        rospy.loginfo_throttle(
            1.0, "STABLE_CONE_CLOUD cones=%d points=%d exclusion=%.2f",
            len(tracks), len(points), self.cone_radius)

    def request_costmap_clear(self, reason):
        now = time.monotonic()
        with self.lock:
            if (self.clear_in_progress or
                    now - self.last_clear_request < self.clear_cooldown):
                return
            self.clear_in_progress = True
            self.last_clear_request = now
        worker = threading.Thread(target=self.clear_costmaps, args=(reason,))
        worker.daemon = True
        worker.start()

    def clear_costmaps(self, reason):
        try:
            rospy.wait_for_service("/move_base/clear_costmaps", timeout=1.0)
            rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
            rospy.logwarn("STABLE_CONE_COSTMAP_CLEARED reason=%s", reason)
        except Exception as exc:
            rospy.logwarn("stable cone clear_costmaps unavailable: %s", exc)
        finally:
            with self.lock:
                self.clear_in_progress = False

    def shutdown(self):
        try:
            self.publish_timer.shutdown()
        except Exception:
            pass
        header = Header(stamp=rospy.Time.now(), frame_id=self.map_frame)
        self.cloud_pub.publish(pc2.create_cloud_xyz32(header, []))


if __name__ == "__main__":
    StableConeMapper()
    rospy.spin()
