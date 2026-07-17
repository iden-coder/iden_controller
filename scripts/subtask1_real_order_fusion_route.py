#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import shlex
import subprocess
import threading
import time

import rospy
from std_msgs.msg import Int32, String


STATE_WAITING_VOICE = 0
STATE_NAV_AND_SCAN = 1
STATE_REASONING = 2
STATE_DONE = 3
STATE_POST_ROUTE = 4
STATE_ERROR = -1


WAREHOUSE_BY_CATEGORY = {
    "食品加工类": "食品加工车间",
    "日用品类": "日用品加工车间",
    "电子产品类": "电子产品生产车间",
}


POST_ROUTE_POINTS = [
    ("start", -0.85, -1.41, -1.5707963268),
    ("d1", -1.63, -2.57, 3.1415926536),
    ("d2", 0.41, -1.60, 1.5707963268),
    ("d3", 2.54, -2.81, -1.5707963268),
    ("d4", 0.373, -3.50, -1.5707963268),
]


MOJIBAKE_REPLACEMENTS = {
    "灏忛": "小飞",
    "椋熷搧鍔犲伐绫": "食品加工类",
    "椋熷搧绫": "食品类",
    "鏃ョ敤鍝佺被": "日用品类",
    "鐢靛瓙浜у搧绫": "电子产品类",
    "鐢佃剳": "电脑",
    "鎵嬫満": "手机",
    "鎵嬫机": "手机",
    "鑺墖": "芯片",
    "澶х背": "大米",
    "鐚倝": "猪肉",
    "姣涘肪": "毛巾",
    "閿洏": "键盘",
    "棣欒晧": "香蕉",
    "妫夎": "棉被",
}


def normalize_text(text):
    if text is None:
        return ""
    return repair_mojibake_text(str(text).strip())


def repair_mojibake_text(text):
    if text is None:
        return ""
    text = str(text)
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    suspicious = ("鐢", "佃", "剳", "鎵", "澶", "姣", "涘", "肪", "鐚", "灏", "椋", "鏃")
    if any(token in text for token in suspicious):
        for encoding in ("gbk", "latin1"):
            try:
                repaired = text.encode(encoding, errors="strict").decode("utf-8")
            except Exception:
                continue
            if repaired and repaired != text:
                for bad, good in MOJIBAKE_REPLACEMENTS.items():
                    repaired = repaired.replace(bad, good)
                return repaired
    return text


def fix_mojibake(value):
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [fix_mojibake(item) for item in value]
    if isinstance(value, dict):
        return {key: fix_mojibake(item) for key, item in value.items()}
    return value


def as_bool(value):
    if isinstance(value, bool):
        return value
    return normalize_text(value).lower() in ("1", "true", "yes", "on")


def clean_secret(value):
    text = normalize_text(value)
    if text in ("''", '""', "none", "None", "null"):
        return ""
    return text


def first_json_object(text):
    text = normalize_text(text)
    if not text:
        return None
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


class Subtask1RealOrderFusion(object):
    def __init__(self):
        rospy.init_node("subtask1_real_order_fusion_route")

        self.voice_topic = rospy.get_param("~voice_topic", "/factory/voice_raw_text")
        self.qr_summary_topic = rospy.get_param("~qr_summary_topic", "/qr_room_scan_results")
        self.qr_item_topic = rospy.get_param("~qr_item_topic", "/qr_room_scan_item")
        self.qr_scan_control_topic = rospy.get_param(
            "~qr_scan_control_topic", "/qr_room_scan_control")
        self.tts_topic = rospy.get_param("~tts_topic", "/factory/tts_text")
        self.result_topic = rospy.get_param("~result_topic", "/factory/subtask1_result")
        self.compat_target_topic = rospy.get_param(
            "~compat_target_topic", "/factory/target_warehouses")
        self.state_topic = rospy.get_param("~state_topic", "/factory/task_state")

        self.required_item_count = int(rospy.get_param("~required_item_count", 3))
        self.streaming_qr_decision = as_bool(rospy.get_param(
            "~streaming_qr_decision", True))
        self.auto_start_nav = as_bool(rospy.get_param("~auto_start_nav", False))
        self.require_wake_before_order = as_bool(rospy.get_param(
            "~require_wake_before_order", True))
        self.require_target_category = as_bool(rospy.get_param(
            "~require_target_category", True))
        self.rearm_asr_on_incomplete_voice = as_bool(rospy.get_param(
            "~rearm_asr_on_incomplete_voice", True))
        self.asr_node_name = rospy.get_param("~asr_node_name", "/cloud_asr_test2")
        self.asr_rearm_delay_s = float(rospy.get_param("~asr_rearm_delay_s", 3.0))
        self.asr_tts_start_timeout_s = float(rospy.get_param(
            "~asr_tts_start_timeout_s", max(6.0, self.asr_rearm_delay_s)))
        self.asr_tts_finish_timeout_s = float(rospy.get_param(
            "~asr_tts_finish_timeout_s", 20.0))
        self.asr_respawn_timeout_s = float(rospy.get_param(
            "~asr_respawn_timeout_s", 8.0))
        self.default_voice_text = rospy.get_param("~default_voice_text", "")
        self.nav_launch_pkg = rospy.get_param("~nav_launch_pkg", "iden_controller")
        self.nav_launch_file = rospy.get_param(
            "~nav_launch_file", "global_first_graph_nav_qr_room.launch")
        self.nav_roslaunch_args = rospy.get_param("~nav_roslaunch_args", "")
        self.nav_start_delay_s = float(rospy.get_param("~nav_start_delay_s", 0.5))
        self.post_route_enabled = as_bool(rospy.get_param("~post_route_enabled", True))
        self.post_route_launch_pkg = rospy.get_param(
            "~post_route_launch_pkg", "iden_controller")
        self.post_route_launch_file = rospy.get_param(
            "~post_route_launch_file", "global_first_graph_nav_2249fcf.launch")
        self.post_route_wait_after_tts_s = float(rospy.get_param(
            "~post_route_wait_after_tts_s", 4.0))
        self.post_route_goal_timeout_s = float(rospy.get_param(
            "~post_route_goal_timeout_s", 180.0))
        self.post_route_hold_after_goal_s = float(rospy.get_param(
            "~post_route_hold_after_goal_s", 0.8))
        self.post_route_status_topic = rospy.get_param(
            "~post_route_status_topic", "/global_first_graph_nav/status")
        self.post_route_node_name = rospy.get_param(
            "~post_route_node_name", "/global_first_graph_nav")
        self.post_route_map_yaml = rospy.get_param("~post_route_map_yaml", "")

        self.use_spark = as_bool(rospy.get_param("~use_spark", True))
        self.require_spark_decision = as_bool(rospy.get_param(
            "~require_spark_decision", True))
        self.spark_url = rospy.get_param(
            "~spark_url", "https://spark-api-open.xf-yun.com/x2/chat/completions")
        self.spark_model = rospy.get_param("~spark_model", "spark-x")
        self.spark_api_key = clean_secret(rospy.get_param(
            "~spark_api_key", os.environ.get("SPARK_API_KEY", "")))
        self.spark_api_secret = clean_secret(rospy.get_param(
            "~spark_api_secret", os.environ.get("SPARK_API_SECRET", "")))
        self.spark_timeout_s = float(rospy.get_param("~spark_timeout_s", 18.0))

        self.lock = threading.Lock()
        self.asr_rearm_lock = threading.Lock()
        self.asr_rearm_in_progress = False
        self.voice_text = ""
        self.target_category = ""
        self.qr_items = []
        self.qr_evidence = []
        self.qr_summary = None
        self.nav_process = None
        self.post_route_process = None
        self.nav_started = False
        self.decision_done = False
        self.stream_decision_in_progress = False
        self.qr_scan_finished = False
        self.processed_qr_keys = set()
        self.post_route_started = False
        self.awakened = False
        self.last_reject_prompt_time = 0.0
        self.latest_nav_status = ""

        self.tts_pub = rospy.Publisher(self.tts_topic, String, queue_size=10)
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=10, latch=True)
        self.compat_pub = rospy.Publisher(
            self.compat_target_topic, String, queue_size=10, latch=True)
        self.state_pub = rospy.Publisher(
            self.state_topic, Int32, queue_size=10, latch=True)
        self.qr_control_pub = rospy.Publisher(
            self.qr_scan_control_topic, String, queue_size=5)

        rospy.Subscriber(self.voice_topic, String, self.voice_callback, queue_size=5)
        rospy.Subscriber(self.qr_summary_topic, String, self.qr_summary_callback, queue_size=3)
        rospy.Subscriber(self.qr_item_topic, String, self.qr_item_callback, queue_size=10)
        rospy.Subscriber(
            self.post_route_status_topic, String, self.nav_status_callback, queue_size=10)

        self.publish_state(STATE_WAITING_VOICE)
        rospy.loginfo("subtask1 real-only fusion ready; waiting voice on %s", self.voice_topic)

        if self.auto_start_nav:
            seed_text = self.default_voice_text or "小飞小飞，前往物品领取区，取得目标货品放置在对应仓库"
            rospy.Timer(rospy.Duration(0.8), lambda _event: self.accept_voice(seed_text),
                        oneshot=True)

    def publish_state(self, state):
        self.state_pub.publish(Int32(data=int(state)))

    def nav_status_callback(self, msg):
        self.latest_nav_status = normalize_text(msg.data).lower()

    def voice_callback(self, msg):
        text = normalize_text(msg.data)
        if not text:
            return
        self.accept_voice(text)

    def accept_voice(self, text):
        text = normalize_text(text)
        category = self.extract_target_category(text)
        wake = self.has_wake_word(text)

        if self.require_wake_before_order and not self.awakened:
            if wake and not category:
                rospy.loginfo("wake word accepted; waiting for target category")
                self.awakened = True
                self.publish_state(STATE_WAITING_VOICE)
                self.prompt_incomplete_order(force=True)
                self.rearm_asr_if_needed()
                return
            if not wake:
                rospy.logwarn("voice ignored before wake word: %s", text)
                self.publish_state(STATE_WAITING_VOICE)
                return

        if self.require_target_category and not category:
            rospy.logwarn(
                "voice ignored because target category is missing; waiting full order: %s",
                text)
            self.publish_state(STATE_WAITING_VOICE)
            self.prompt_incomplete_order()
            self.rearm_asr_if_needed()
            return

        with self.lock:
            if self.voice_text:
                rospy.logwarn("voice already accepted; ignoring later voice: %s", text)
                return
            self.voice_text = text
            self.target_category = category
        rospy.loginfo("voice command accepted: %s", text)
        if self.target_category:
            rospy.loginfo("real target category: %s", self.target_category)
        else:
            rospy.logwarn("target category not found in voice; Spark will try later")
        self.speak("已接收任务，开始前往物品领取区。")
        self.start_nav_if_needed()
        self.try_decide()

    def has_wake_word(self, text):
        cleaned = normalize_text(text).replace(" ", "")
        wake_words = (
            "\u5c0f\u98de\u5c0f\u98de",
            "\u5c0f\u98de",
            "\u5c0f\u8f89\u5c0f\u8f89",
            "\u5c0f\u8f89",
        )
        return any(word in cleaned for word in wake_words)

    def prompt_incomplete_order(self, force=False):
        now = time.time()
        if force or now - self.last_reject_prompt_time > 4.0:
            self.last_reject_prompt_time = now
            self.speak("\u6211\u5728\uff0c\u8bf7\u8bf4\u660e\u9700\u8981\u9886\u53d6\u7684\u8d27\u54c1\u5927\u7c7b\u3002")

    def extract_target_category(self, text):
        cleaned = normalize_text(text)
        patterns = [
            (r"食品加工|食品大类|食品类|食品|食物|水果|肉|米|粮", "食品加工类"),
            (r"日用品加工|日用品大类|日用品类|日用品|生活用品|毛巾|牙刷|棉被|衣物", "日用品类"),
            (r"电子产品生产|电子产品大类|电子产品类|电子产品|电子|电脑|手机|芯片", "电子产品类"),
        ]
        for pattern, category in patterns:
            if re.search(pattern, cleaned):
                return category
        return ""

    def start_nav_if_needed(self):
        with self.lock:
            if self.nav_started:
                return
            self.nav_started = True
        self.publish_state(STATE_NAV_AND_SCAN)
        threading.Thread(target=self.start_nav_thread, daemon=True).start()

    def start_nav_thread(self):
        rospy.sleep(self.nav_start_delay_s)
        command = ["roslaunch", self.nav_launch_pkg, self.nav_launch_file]
        if self.nav_roslaunch_args:
            command.extend(shlex.split(self.nav_roslaunch_args))
        rospy.loginfo("starting navigation and QR scan: %s", " ".join(command))
        try:
            self.nav_process = subprocess.Popen(command)
        except Exception as exc:
            rospy.logerr("failed to start nav launch: %s", exc)
            self.publish_error("导航启动失败")

    def rearm_asr_if_needed(self):
        if not self.rearm_asr_on_incomplete_voice:
            return
        with self.asr_rearm_lock:
            if self.asr_rearm_in_progress:
                rospy.logwarn("ASR rearm already in progress; duplicate request ignored")
                return
            self.asr_rearm_in_progress = True
        threading.Thread(target=self.rearm_asr_thread, daemon=True).start()

    @staticmethod
    def tts_process_active():
        """Return true while the shared ASR node is synthesizing or playing TTS."""
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open("/proc/{}/cmdline".format(entry), "rb") as handle:
                        command = handle.read().replace(b"\x00", b" ")
                except (IOError, OSError):
                    continue
                if b"xf_tts_stable.py" in command:
                    return True
                if b"aplay" in command and b"tts_result.pcm" in command:
                    return True
        except (IOError, OSError):
            return False
        return False

    def wait_for_prompt_playback(self):
        start = time.time()
        finish_deadline = start + max(
            self.asr_tts_start_timeout_s, self.asr_tts_finish_timeout_s)
        seen_tts = False
        quiet_since = None

        rospy.loginfo("ASR_REARM_WAIT_TTS waiting for spoken prompt to finish")
        while not rospy.is_shutdown() and time.time() < finish_deadline:
            now = time.time()
            active = self.tts_process_active()
            if active:
                seen_tts = True
                quiet_since = None
            elif seen_tts:
                if quiet_since is None:
                    quiet_since = now
                elif now - quiet_since >= 0.45:
                    rospy.loginfo("ASR_REARM_TTS_FINISHED prompt playback completed")
                    return
            elif now - start >= self.asr_tts_start_timeout_s:
                rospy.logwarn(
                    "ASR_REARM_TTS_NOT_SEEN after %.1fs; recovering ASR anyway",
                    now - start)
                return
            rospy.sleep(0.1)

        rospy.logwarn(
            "ASR_REARM_TTS_TIMEOUT after %.1fs; recovering ASR",
            time.time() - start)

    def wait_for_asr_respawn(self):
        deadline = time.time() + max(1.0, self.asr_respawn_timeout_s)
        while not rospy.is_shutdown() and time.time() < deadline:
            try:
                output = subprocess.check_output(
                    ["rosnode", "list"], universal_newlines=True)
                if self.asr_node_name in output.splitlines():
                    rospy.loginfo(
                        "ASR_REARM_READY %s; please state the target category",
                        self.asr_node_name)
                    return True
            except Exception:
                pass
            rospy.sleep(0.2)
        rospy.logwarn("ASR_REARM_RESPAWN_TIMEOUT node=%s", self.asr_node_name)
        return False

    def rearm_asr_thread(self):
        try:
            self.wait_for_prompt_playback()
            if rospy.is_shutdown():
                return
            rospy.logwarn("rearming ASR node after spoken prompt")
            subprocess.call(["rosnode", "kill", self.asr_node_name])
            self.wait_for_asr_respawn()
        except Exception as exc:
            rospy.logwarn("failed to rearm ASR node %s: %s", self.asr_node_name, exc)
        finally:
            with self.asr_rearm_lock:
                self.asr_rearm_in_progress = False

    def qr_summary_callback(self, msg):
        summary = first_json_object(msg.data)
        if not isinstance(summary, dict):
            rospy.logwarn("invalid QR summary ignored: %s", msg.data[:120])
            return
        items = self.extract_items_from_summary(summary)
        evidence = self.build_qr_evidence(summary)
        status = normalize_text(summary.get("status")).lower()
        with self.lock:
            if self.decision_done:
                return
            self.qr_summary = summary
            self.qr_scan_finished = status in (
                "complete", "partial", "stopped_by_decision", "done", "finished")
            self.qr_items = items
            self.qr_evidence = evidence
        rospy.loginfo("QR summary accepted: %s", items)
        self.try_decide()

    def qr_item_callback(self, msg):
        if not self.streaming_qr_decision:
            return
        event = first_json_object(msg.data)
        if not isinstance(event, dict):
            rospy.logwarn("invalid QR item ignored: %s", msg.data[:120])
            return
        result = self.extract_result_from_item_event(event)
        if not isinstance(result, dict):
            rospy.logwarn("QR item event has no result: %s", msg.data[:160])
            return
        items = self.extract_items_from_wall_result(result)
        evidence = self.build_qr_evidence({"wall_results": [result]})
        key = self.qr_result_key(result)
        with self.lock:
            if self.decision_done:
                return
            if key in self.processed_qr_keys:
                return
            self.processed_qr_keys.add(key)
            self.merge_qr_observation_locked(items, evidence)
            voice_text = self.voice_text
            category = self.target_category
            if self.stream_decision_in_progress:
                rospy.loginfo("STREAM_QR_ITEM_QUEUED items=%s", items)
                return
            self.stream_decision_in_progress = True
        rospy.loginfo("STREAM_QR_ITEM wall=%s items=%s", result.get("wall_index"), items)
        self.publish_state(STATE_REASONING)
        threading.Thread(
            target=self.stream_decide_thread,
            args=(voice_text, category, items, evidence),
            daemon=True).start()

    def extract_result_from_item_event(self, event):
        for key in ("result", "wall_result", "qr", "item"):
            value = event.get(key)
            if isinstance(value, dict):
                return value
        if "parsed" in event or "raw" in event:
            return event
        return None

    def qr_result_key(self, result):
        raw = normalize_text(result.get("raw"))
        wall = result.get("wall_index")
        parsed = result.get("parsed")
        parsed_text = json.dumps(parsed, ensure_ascii=False, sort_keys=True) if isinstance(parsed, dict) else ""
        return "{}|{}|{}".format(wall, raw, parsed_text)

    def merge_qr_observation_locked(self, items, evidence):
        for item in items:
            if item and item not in self.qr_items:
                self.qr_items.append(item)
        for entry in evidence:
            key = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            exists = False
            for old in self.qr_evidence:
                old_key = json.dumps(old, ensure_ascii=False, sort_keys=True)
                if old_key == key:
                    exists = True
                    break
            if not exists:
                self.qr_evidence.append(entry)

    def extract_items_from_summary(self, summary):
        items = []
        for result in self.iter_qr_results(summary):
            for item in self.extract_items_from_wall_result(result):
                if item and item not in items:
                    items.append(item)
        return items

    def iter_qr_results(self, summary):
        seen = set()
        for key in ("wall_results", "results", "detections", "qrs", "items"):
            values = summary.get(key, [])
            if isinstance(values, dict):
                values = list(values.values())
            if not isinstance(values, list):
                continue
            for result in values:
                if not isinstance(result, dict):
                    continue
                stamp = result.get("stamp")
                raw = normalize_text(result.get("raw"))
                identity = (result.get("wall_index"), stamp, raw)
                if identity in seen:
                    continue
                seen.add(identity)
                yield result

    def extract_item_from_wall_result(self, result):
        items = self.extract_items_from_wall_result(result)
        return items[0] if items else ""

    def extract_items_from_wall_result(self, result):
        if not isinstance(result, dict):
            return []
        candidates = []
        parsed = result.get("parsed")
        if isinstance(parsed, dict):
            candidates.extend(self.extract_items_from_any(parsed.get("json")))
            candidates.extend(self.extract_items_from_any(parsed.get("text")))
        candidates.extend(self.extract_items_from_any(result.get("raw")))
        cleaned = []
        for item in candidates:
            item = self.clean_candidate_item(item)
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned

    def extract_item_from_any(self, value):
        items = self.extract_items_from_any(value)
        return items[0] if items else ""

    def extract_items_from_any(self, value):
        if value is None:
            return []
        if isinstance(value, dict):
            items = []
            for key in ("result", "name", "item", "goods", "product", "货品", "物品", "名称"):
                item = normalize_text(value.get(key))
                if item:
                    items.append(item)
            for nested in value.values():
                items.extend(self.extract_items_from_any(nested))
            return items
        if isinstance(value, list):
            items = []
            for nested in value:
                items.extend(self.extract_items_from_any(nested))
            return items
        text = normalize_text(value)
        parsed = first_json_object(text)
        if parsed is not None and parsed is not value:
            parsed_items = self.extract_items_from_any(parsed)
            if parsed_items:
                return parsed_items
        if text.startswith("http://") or text.startswith("https://"):
            return []
        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_+-]+", text)
        return tokens

    def clean_candidate_item(self, item):
        item = normalize_text(item)
        if not item:
            return ""
        item = item.strip(" \t\r\n:：,，;；。\"'[]{}()（）")
        if not item or item.lower().startswith(("http", "www")):
            return ""
        if item.lower() in ("code", "result", "url", "json", "text", "none", "null", "true", "false"):
            return ""
        if item.isdigit():
            return ""
        if len(item) > 24:
            return ""
        return item

    def category_hint_from_qr_result(self, result):
        if not isinstance(result, dict):
            return ""
        texts = [normalize_text(result.get("raw"))]
        parsed = result.get("parsed")
        if isinstance(parsed, dict):
            texts.append(normalize_text(parsed.get("url")))
            texts.append(normalize_text(parsed.get("type")))
        text = " ".join(part for part in texts if part).lower()
        if "qrcode/food" in text or "/food" in text:
            return "食品加工类"
        if "qrcode/daily" in text or "/daily" in text:
            return "日用品类"
        if "qrcode/electronic" in text or "/electronic" in text or "/electronics" in text:
            return "电子产品类"
        return ""

    def build_qr_evidence(self, summary):
        evidence = []
        for result in self.iter_qr_results(summary):
            parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
            entry = {
                "wall_index": result.get("wall_index"),
                "raw": normalize_text(result.get("raw")),
                "category_hint": self.category_hint_from_qr_result(result),
                "parsed_type": normalize_text(parsed.get("type")),
                "url": normalize_text(parsed.get("url")),
                "json": fix_mojibake(parsed.get("json")),
                "text": normalize_text(parsed.get("text")),
                "error": normalize_text(parsed.get("error")),
                "candidate_items": self.extract_items_from_wall_result(result),
            }
            evidence.append(entry)
        return evidence

    def try_decide(self):
        with self.lock:
            if self.decision_done:
                return
            voice_text = self.voice_text
            category = self.target_category
            items = list(self.qr_items)
            evidence = list(self.qr_evidence)
            scan_finished = self.qr_scan_finished
            stream_busy = self.stream_decision_in_progress
        if not voice_text:
            return
        if stream_busy:
            rospy.loginfo_throttle(
                5.0, "waiting active single-QR Spark decision before final decision")
            return
        if len(items) < self.required_item_count:
            rospy.loginfo_throttle(
                5.0,
                "waiting QR items: %d/%d; Spark decision is disabled until all required QR items are collected",
                len(items), self.required_item_count)
            if scan_finished:
                rospy.logwarn_throttle(
                    5.0,
                    "QR summary was partial; waiting for scanner to keep rotating until %d valid QR items",
                    self.required_item_count)
            return

        with self.lock:
            if self.decision_done:
                return
            self.decision_done = True
        self.publish_state(STATE_REASONING)
        threading.Thread(target=self.decide_thread, args=(voice_text, category, items, evidence),
                         daemon=True).start()

    def stream_decide_thread(self, voice_text, category, items, evidence=None):
        evidence = evidence or []
        try:
            if not voice_text:
                rospy.logwarn("STREAM_QR_WAITING_VOICE items=%s", items)
                self.qr_control_pub.publish(String(data="continue"))
                return
            if not items:
                rospy.logwarn("STREAM_QR_EMPTY_ITEM ignored")
                self.qr_control_pub.publish(String(data="continue"))
                return
            if not category:
                category = self.extract_target_category(voice_text)
            if not category:
                category = self.infer_category_with_spark_or_local(voice_text, items)
            if not category:
                rospy.logwarn("STREAM_QR_NO_TARGET_CATEGORY voice=%s", voice_text)
                self.qr_control_pub.publish(String(data="continue"))
                return
            if self.require_spark_decision and not self.spark_ready():
                self.publish_error("星火大模型未配置，不能进行货品筛选")
                return

            rospy.loginfo("STREAM_QR_SPARK_CHECK category=%s items=%s", category, items)
            selected_item = self.select_item_with_spark(category, items, evidence)
            if not selected_item:
                rospy.loginfo("STREAM_QR_NO_MATCH category=%s items=%s; continue scanning",
                              category, items)
                self.qr_control_pub.publish(String(data="continue"))
                self.publish_state(STATE_NAV_AND_SCAN)
                return

            with self.lock:
                if self.decision_done:
                    return
                self.decision_done = True
                all_items = list(self.qr_items)
            rospy.logwarn("STREAM_QR_MATCH_STOP selected_item=%s; stopping QR scan",
                          selected_item)
            self.qr_control_pub.publish(String(data="stop"))
            self.finish_success(voice_text, category, selected_item, all_items)
        finally:
            with self.lock:
                self.stream_decision_in_progress = False
                should_try_final = self.qr_scan_finished and not self.decision_done
            if should_try_final:
                self.try_decide()

    def decide_thread(self, voice_text, category, items, evidence=None):
        evidence = evidence or []
        if not category:
            category = self.extract_target_category(voice_text)
        if not category:
            category = self.infer_category_with_spark_or_local(voice_text, items)
        if not category:
            self.publish_error("没有识别到目标大类")
            return

        if self.require_spark_decision and not self.spark_ready():
            self.publish_error("星火大模型未配置，不能进行货品筛选")
            return

        selected_item = self.select_item_with_spark(category, items, evidence)
        if selected_item:
            self.finish_success(voice_text, category, selected_item, items)
            return
        if not selected_item:
            self.publish_error("星火大模型未能从二维码中选出目标货品")
            return

        warehouse = WAREHOUSE_BY_CATEGORY.get(category, "未知车间")
        text = "取得{}属于{}应放置在{}。".format(selected_item, category, warehouse)
        payload = {
            "status": "success",
            "voice_text": voice_text,
            "target_category": category,
            "selected_item": selected_item,
            "target_warehouse": warehouse,
            "scanned_items": items,
            "broadcast_text": text,
            "sim_task_ignored": True,
            "stamp": time.time(),
        }
        rospy.loginfo("subtask1 decision: %s", json.dumps(payload, ensure_ascii=False))
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.compat_pub.publish(String(data=json.dumps({
            "real_item": selected_item,
            "real_category": category,
            "real_warehouse": warehouse,
            "sim_ignored": True,
        }, ensure_ascii=False)))
        self.speak(text)
        if self.post_route_enabled:
            self.start_post_route_after_tts()
        else:
            self.publish_state(STATE_DONE)

    def finish_success(self, voice_text, category, selected_item, items):
        warehouse = WAREHOUSE_BY_CATEGORY.get(category, "未知车间")
        text = "取得{}属于{}应放置在{}。".format(selected_item, category, warehouse)
        payload = {
            "status": "success",
            "voice_text": voice_text,
            "target_category": category,
            "selected_item": selected_item,
            "target_warehouse": warehouse,
            "scanned_items": items,
            "broadcast_text": text,
            "sim_task_ignored": True,
            "stamp": time.time(),
        }
        rospy.loginfo("subtask1 decision: %s", json.dumps(payload, ensure_ascii=False))
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.compat_pub.publish(String(data=json.dumps({
            "real_item": selected_item,
            "real_category": category,
            "real_warehouse": warehouse,
            "sim_ignored": True,
        }, ensure_ascii=False)))
        self.speak(text)
        if self.post_route_enabled:
            self.start_post_route_after_tts()
        else:
            self.publish_state(STATE_DONE)

    def start_post_route_after_tts(self):
        with self.lock:
            if self.post_route_started:
                return
            self.post_route_started = True
        threading.Thread(target=self.post_route_thread, daemon=True).start()

    def post_route_thread(self):
        self.publish_state(STATE_POST_ROUTE)
        rospy.loginfo(
            "waiting %.1fs for TTS before post-route navigation",
            self.post_route_wait_after_tts_s)
        rospy.sleep(max(0.0, self.post_route_wait_after_tts_s))

        for name, x, y, yaw in POST_ROUTE_POINTS:
            if rospy.is_shutdown():
                return
            ok = self.navigate_post_point(name, x, y, yaw)
            if not ok:
                self.publish_error("后续点位{}导航超时或失败".format(name))
                return
            rospy.sleep(max(0.0, self.post_route_hold_after_goal_s))

        done_text = "后续点位导航完成。"
        self.speak(done_text)
        payload = {
            "status": "post_route_complete",
            "points": [pt[0] for pt in POST_ROUTE_POINTS],
            "stamp": time.time(),
        }
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.publish_state(STATE_DONE)

    def navigate_post_point(self, name, x, y, yaw):
        self.kill_post_nav_node()
        self.latest_nav_status = ""
        command = [
            "roslaunch",
            self.post_route_launch_pkg,
            self.post_route_launch_file,
            "goal_x:={:.3f}".format(x),
            "goal_y:={:.3f}".format(y),
            "goal_yaw:={:.10f}".format(yaw),
        ]
        if self.post_route_map_yaml:
            command.append("map_yaml:={}".format(self.post_route_map_yaml))
        rospy.logwarn("post-route navigating to %s: x=%.3f y=%.3f yaw=%.2f",
                      name, x, y, yaw)
        try:
            self.post_route_process = subprocess.Popen(command)
        except Exception as exc:
            rospy.logerr("failed to launch post-route nav for %s: %s", name, exc)
            return False

        start = rospy.Time.now()
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            status = self.latest_nav_status
            if "goal reached" in status:
                rospy.logwarn("post-route point %s reached", name)
                self.kill_post_nav_node()
                return True
            if (rospy.Time.now() - start).to_sec() > self.post_route_goal_timeout_s:
                rospy.logerr("post-route point %s timeout; last status=%s", name, status)
                self.kill_post_nav_node()
                return False
            rate.sleep()
        return False

    def kill_post_nav_node(self):
        try:
            subprocess.call(["rosnode", "kill", self.post_route_node_name])
        except Exception:
            pass
        self.post_route_process = None

    def infer_category_with_spark_or_local(self, voice_text, items):
        category = self.extract_target_category(voice_text)
        if category:
            return category
        if not self.spark_ready():
            return ""
        prompt = (
            "只处理现实环境任务，忽略仿真环境。"
            "请从语音命令中提取目标大类，只能返回以下三者之一："
            "食品加工类、日用品类、电子产品类。"
            "如果无法判断，返回空字符串。"
            "语音命令：{}；已扫描货品：{}。"
        ).format(voice_text, "、".join(items))
        data = self.call_spark(prompt, max_tokens=80)
        text = self.extract_spark_content(data)
        return self.normalize_category(text)

    def select_item_with_spark(self, category, items, evidence=None):
        if not self.spark_ready():
            return ""
        evidence = evidence or []
        candidates = []
        for item in items:
            clean = self.clean_candidate_item(item)
            if clean and clean not in candidates:
                candidates.append(clean)
        rospy.loginfo("SPARK_DECISION_START category=%s candidates=%s evidence_count=%d",
                      category, candidates, len(evidence))
        prompt_payload = {
            "target_category": category,
            "candidate_items": candidates,
            "qr_evidence": evidence,
            "allow_no_match": True,
            "no_match_output": {
                "selected_item": "",
                "reason": "no candidate belongs to target_category",
            },
            "rules": [
                "CRITICAL: choose selected_item only if it clearly belongs to target_category; otherwise selected_item must be empty.",
                "QR category_hint or URL path is authoritative: /food means 食品加工类, /daily means 日用品类, /electronic means 电子产品类.",
                "If category_hint conflicts with target_category, return {\"selected_item\":\"\",\"reason\":\"category mismatch\"}.",
                "日用品类 means non-electronic daily household, textile, or cleaning goods. Computer peripherals, electronic devices, and electronic components belong to 电子产品类.",
                "只处理现实环境任务，忽略仿真环境。",
                "必须根据目标大类从二维码候选货品中选择一个。",
                "优先从 candidate_items 中选择；如果 candidate_items 是乱码或不完整，可以参考 qr_evidence 中的 raw/json/text/url。",
                "不要创造二维码中没有出现过的货品。",
                "只返回 JSON，不要解释。",
            ],
            "required_output": {
                "selected_item": "从二维码中选出的货品原名或修正后的中文名",
                "reason": "极短理由",
            },
        }
        prompt = (
            "你是无人工厂现实环境货品筛选节点。请读取以下JSON任务数据，"
            "从二维码候选货品中选择属于目标大类的唯一货品。"
            "输出必须是严格JSON，格式为 {\"selected_item\":\"货品名称\",\"reason\":\"简短理由\"}。\n"
            + json.dumps(prompt_payload, ensure_ascii=False)
            + "\nIf no candidate item belongs to target_category, return exactly "
              "{\"selected_item\":\"\",\"reason\":\"no match\"}."
        )
        for attempt in range(2):
            data = self.call_spark(prompt, max_tokens=180)
            content = self.extract_spark_content(data)
            selected = self.parse_selected_item_from_spark(content, candidates)
            if selected and self.selected_item_conflicts_with_evidence(
                    category, selected, evidence):
                rospy.logwarn(
                    "SPARK_DECISION_REJECTED_BY_QR_CATEGORY selected_item=%s category=%s raw_content=%s",
                    selected, category, content[:200])
                selected = ""
            if selected:
                rospy.loginfo("SPARK_DECISION_OK selected_item=%s raw_content=%s",
                              selected, content[:200])
                return selected
            rospy.logwarn("Spark response could not be parsed on attempt %d: %s",
                          attempt + 1, content[:200])
            prompt += "\n上一次返回无法解析。请只返回JSON，例如：{\"selected_item\":\"香蕉\",\"reason\":\"属于食品加工类\"}。"
        return ""

    def selected_item_conflicts_with_evidence(self, category, selected_item, evidence):
        category = self.normalize_category(category)
        selected_item = self.clean_candidate_item(selected_item)
        if not category or not selected_item:
            return False
        hints = []
        for entry in evidence or []:
            if not isinstance(entry, dict):
                continue
            candidate_items = entry.get("candidate_items") or []
            matched_item = False
            for item in candidate_items:
                item = self.clean_candidate_item(item)
                if item and (selected_item == item or selected_item in item or item in selected_item):
                    matched_item = True
                    break
            if not matched_item:
                continue
            hint = self.normalize_category(entry.get("category_hint"))
            if hint:
                hints.append(hint)
        return bool(hints and category not in hints)

    def spark_response_says_no_match(self, content, obj=None):
        text = normalize_text(content).lower()
        markers = (
            "no match", "no candidate", "none", "category mismatch",
            "not belong", "does not belong", "不属于", "不匹配",
            "没有符合", "无匹配", "无法匹配", "不是目标", "类别不符",
        )
        if any(marker in text for marker in markers):
            return True
        if isinstance(obj, dict):
            for key in ("match", "matched", "is_match"):
                value = obj.get(key)
                if value is False:
                    return True
                if isinstance(value, str) and value.strip().lower() in ("false", "no", "0"):
                    return True
        return False

    def parse_selected_item_from_spark(self, content, candidates):
        content = normalize_text(content)
        if not content:
            return ""
        obj = first_json_object(content)
        values = []
        if isinstance(obj, dict):
            explicit_selection_key_seen = False
            for key in ("selected_item", "item", "name", "goods", "product", "货品", "物品", "result"):
                if key not in obj:
                    continue
                explicit_selection_key_seen = True
                value = normalize_text(obj.get(key))
                if value:
                    values.append(value)
            index_value = obj.get("index")
            if isinstance(index_value, int) and 0 <= index_value < len(candidates):
                values.append(candidates[index_value])
            elif isinstance(index_value, str) and index_value.isdigit():
                idx = int(index_value)
                if 0 <= idx < len(candidates):
                    values.append(candidates[idx])
            if self.spark_response_says_no_match(content, obj):
                return ""
            if explicit_selection_key_seen and not values:
                return ""
        elif self.spark_response_says_no_match(content):
            return ""
        else:
            values.append(content)
        for value in values:
            value = self.clean_candidate_item(value)
            if not value:
                continue
            for item in candidates:
                if value == item:
                    return item
            for item in candidates:
                if value in item or item in value:
                    return item
        if isinstance(obj, dict):
            return ""
        for item in candidates:
            if item and item in content:
                return item
        return ""

    def select_item_locally(self, category, items):
        raise RuntimeError("local item-category fallback is disabled; Spark decision is required")

    def guess_item_category_by_keyword(self, item):
        raise RuntimeError("keyword fallback is disabled; Spark decision is required")

    def normalize_category(self, text):
        text = normalize_text(text)
        if not text:
            return ""
        for category in WAREHOUSE_BY_CATEGORY:
            if category in text:
                return category
        return self.extract_target_category(text)

    def spark_ready(self):
        return bool(self.use_spark and self.spark_api_key and self.spark_api_secret)

    def call_spark(self, prompt, max_tokens=120):
        if not self.spark_ready():
            rospy.logwarn("SPARK_CALL_SKIPPED reason=missing_key_or_disabled")
            return None
        try:
            import requests
            headers = {
                "Authorization": "Bearer {}:{}".format(
                    self.spark_api_key, self.spark_api_secret),
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.spark_model,
                "messages": [
                    {"role": "system", "content": "只输出最终答案，不要输出推理过程。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.05,
                "max_tokens": int(max_tokens),
            }
            rospy.loginfo("SPARK_CALL_START model=%s max_tokens=%d prompt_chars=%d",
                          self.spark_model, int(max_tokens), len(prompt))
            started = time.time()
            resp = requests.post(
                self.spark_url, headers=headers, json=payload,
                timeout=self.spark_timeout_s)
            elapsed = time.time() - started
            if resp.status_code != 200:
                rospy.logwarn("SPARK_HTTP_FAIL status=%s elapsed=%.2fs body=%s",
                              resp.status_code, elapsed, resp.text[:200])
                return None
            rospy.loginfo("SPARK_HTTP_OK status=%s elapsed=%.2fs response_chars=%d",
                          resp.status_code, elapsed, len(resp.text))
            return resp.json()
        except Exception as exc:
            rospy.logwarn("SPARK_CALL_EXCEPTION %s", exc)
            return None

    def extract_spark_content(self, data):
        try:
            return normalize_text(data["choices"][0]["message"]["content"])
        except Exception:
            return ""

    def publish_error(self, reason):
        rospy.logerr("subtask1 failed: %s", reason)
        payload = {
            "status": "error",
            "reason": reason,
            "voice_text": self.voice_text,
            "scanned_items": self.qr_items,
            "sim_task_ignored": True,
            "stamp": time.time(),
        }
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.speak(reason + "，请人工确认。")
        self.publish_state(STATE_ERROR)

    def speak(self, text):
        self.tts_pub.publish(String(data=text))


def main():
    node = Subtask1RealOrderFusion()
    rospy.spin()


if __name__ == "__main__":
    main()
