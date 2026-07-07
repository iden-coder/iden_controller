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
STATE_ERROR = -1


WAREHOUSE_BY_CATEGORY = {
    "食品加工类": "食品加工车间",
    "日用品类": "日用品加工车间",
    "电子产品类": "电子产品生产车间",
}


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
}


LOCAL_ITEM_CATEGORY = {
    "苹果": "食品加工类",
    "香蕉": "食品加工类",
    "梨": "食品加工类",
    "橙子": "食品加工类",
    "大米": "食品加工类",
    "面包": "食品加工类",
    "牛奶": "食品加工类",
    "鸡蛋": "食品加工类",
    "猪肉": "食品加工类",
    "牛肉": "食品加工类",
    "鱼": "食品加工类",
    "饼干": "食品加工类",
    "土豆": "食品加工类",
    "番茄": "食品加工类",
    "毛巾": "日用品类",
    "牙刷": "日用品类",
    "牙膏": "日用品类",
    "香皂": "日用品类",
    "肥皂": "日用品类",
    "洗发水": "日用品类",
    "纸巾": "日用品类",
    "棉被": "日用品类",
    "枕头": "日用品类",
    "雨伞": "日用品类",
    "水杯": "日用品类",
    "杯子": "日用品类",
    "T恤": "日用品类",
    "t恤": "日用品类",
    "衬衫": "日用品类",
    "衣服": "日用品类",
    "电脑": "电子产品类",
    "手机": "电子产品类",
    "芯片": "电子产品类",
    "鼠标": "电子产品类",
    "键盘": "电子产品类",
    "耳机": "电子产品类",
    "相机": "电子产品类",
    "平板": "电子产品类",
    "电池": "电子产品类",
    "充电器": "电子产品类",
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
        rospy.init_node("subtask1_real_order_fusion")

        self.voice_topic = rospy.get_param("~voice_topic", "/factory/voice_raw_text")
        self.qr_summary_topic = rospy.get_param("~qr_summary_topic", "/qr_room_scan_results")
        self.tts_topic = rospy.get_param("~tts_topic", "/factory/tts_text")
        self.result_topic = rospy.get_param("~result_topic", "/factory/subtask1_result")
        self.compat_target_topic = rospy.get_param(
            "~compat_target_topic", "/factory/target_warehouses")
        self.state_topic = rospy.get_param("~state_topic", "/factory/task_state")

        self.required_item_count = int(rospy.get_param("~required_item_count", 3))
        self.auto_start_nav = as_bool(rospy.get_param("~auto_start_nav", False))
        self.require_wake_before_order = as_bool(rospy.get_param(
            "~require_wake_before_order", True))
        self.require_target_category = as_bool(rospy.get_param(
            "~require_target_category", True))
        self.rearm_asr_on_incomplete_voice = as_bool(rospy.get_param(
            "~rearm_asr_on_incomplete_voice", True))
        self.asr_node_name = rospy.get_param("~asr_node_name", "/cloud_asr_test2")
        self.asr_rearm_delay_s = float(rospy.get_param("~asr_rearm_delay_s", 3.0))
        self.default_voice_text = rospy.get_param("~default_voice_text", "")
        self.nav_launch_pkg = rospy.get_param("~nav_launch_pkg", "iden_controller")
        self.nav_launch_file = rospy.get_param(
            "~nav_launch_file", "global_first_graph_nav_qr_room.launch")
        self.nav_roslaunch_args = rospy.get_param("~nav_roslaunch_args", "")
        self.nav_start_delay_s = float(rospy.get_param("~nav_start_delay_s", 0.5))

        self.use_spark = as_bool(rospy.get_param("~use_spark", True))
        self.spark_url = rospy.get_param(
            "~spark_url", "https://spark-api-open.xf-yun.com/x2/chat/completions")
        self.spark_model = rospy.get_param("~spark_model", "spark-x")
        self.spark_api_key = clean_secret(rospy.get_param(
            "~spark_api_key", os.environ.get("SPARK_API_KEY", "")))
        self.spark_api_secret = clean_secret(rospy.get_param(
            "~spark_api_secret", os.environ.get("SPARK_API_SECRET", "")))
        self.spark_timeout_s = float(rospy.get_param("~spark_timeout_s", 18.0))

        self.lock = threading.Lock()
        self.voice_text = ""
        self.target_category = ""
        self.qr_items = []
        self.qr_summary = None
        self.nav_process = None
        self.nav_started = False
        self.decision_done = False
        self.awakened = False
        self.last_reject_prompt_time = 0.0

        self.tts_pub = rospy.Publisher(self.tts_topic, String, queue_size=10)
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=10, latch=True)
        self.compat_pub = rospy.Publisher(
            self.compat_target_topic, String, queue_size=10, latch=True)
        self.state_pub = rospy.Publisher(
            self.state_topic, Int32, queue_size=10, latch=True)

        rospy.Subscriber(self.voice_topic, String, self.voice_callback, queue_size=5)
        rospy.Subscriber(self.qr_summary_topic, String, self.qr_summary_callback, queue_size=3)

        self.publish_state(STATE_WAITING_VOICE)
        rospy.loginfo("subtask1 real-only fusion ready; waiting voice on %s", self.voice_topic)

        if self.auto_start_nav:
            seed_text = self.default_voice_text or "小飞小飞，前往物品领取区，取得目标货品放置在对应仓库"
            rospy.Timer(rospy.Duration(0.8), lambda _event: self.accept_voice(seed_text),
                        oneshot=True)

    def publish_state(self, state):
        self.state_pub.publish(Int32(data=int(state)))

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
            rospy.logwarn("target category not found in voice; Spark/local fallback will try later")
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
        threading.Thread(target=self.rearm_asr_thread, daemon=True).start()

    def rearm_asr_thread(self):
        rospy.sleep(max(0.1, self.asr_rearm_delay_s))
        if rospy.is_shutdown():
            return
        rospy.logwarn("rearming ASR node after incomplete voice command")
        try:
            subprocess.call(["rosnode", "kill", self.asr_node_name])
        except Exception as exc:
            rospy.logwarn("failed to rearm ASR node %s: %s", self.asr_node_name, exc)

    def qr_summary_callback(self, msg):
        summary = first_json_object(msg.data)
        if not isinstance(summary, dict):
            rospy.logwarn("invalid QR summary ignored: %s", msg.data[:120])
            return
        items = self.extract_items_from_summary(summary)
        with self.lock:
            self.qr_summary = summary
            self.qr_items = items
        rospy.loginfo("QR summary accepted: %s", items)
        self.try_decide()

    def extract_items_from_summary(self, summary):
        items = []
        for result in summary.get("wall_results", []):
            item = self.extract_item_from_wall_result(result)
            if item and item not in items:
                items.append(item)
        for result in summary.get("results", []):
            item = self.extract_item_from_wall_result(result)
            if item and item not in items:
                items.append(item)
        return items

    def extract_item_from_wall_result(self, result):
        if not isinstance(result, dict):
            return ""
        parsed = result.get("parsed")
        if isinstance(parsed, dict):
            item = self.extract_item_from_any(parsed.get("json"))
            if item:
                return item
            item = self.extract_item_from_any(parsed.get("text"))
            if item:
                return item
        return self.extract_item_from_any(result.get("raw"))

    def extract_item_from_any(self, value):
        if value is None:
            return ""
        if isinstance(value, dict):
            for key in ("result", "name", "item", "goods", "product", "货品", "物品", "名称"):
                item = normalize_text(value.get(key))
                if item:
                    return item
            for nested in value.values():
                item = self.extract_item_from_any(nested)
                if item:
                    return item
            return ""
        if isinstance(value, list):
            for nested in value:
                item = self.extract_item_from_any(nested)
                if item:
                    return item
            return ""
        text = normalize_text(value)
        parsed = first_json_object(text)
        if parsed is not None and parsed is not value:
            item = self.extract_item_from_any(parsed)
            if item:
                return item
        match = re.search(r"[\u4e00-\u9fffA-Za-z0-9]+", text)
        return match.group(0) if match else ""

    def try_decide(self):
        with self.lock:
            if self.decision_done:
                return
            voice_text = self.voice_text
            category = self.target_category
            items = list(self.qr_items)
        if not voice_text:
            return
        if len(items) < self.required_item_count:
            rospy.loginfo_throttle(
                5.0, "waiting QR items: %d/%d", len(items), self.required_item_count)
            return

        with self.lock:
            if self.decision_done:
                return
            self.decision_done = True
        self.publish_state(STATE_REASONING)
        threading.Thread(target=self.decide_thread, args=(voice_text, category, items),
                         daemon=True).start()

    def decide_thread(self, voice_text, category, items):
        if not category:
            category = self.extract_target_category(voice_text)
        if not category:
            category = self.infer_category_with_spark_or_local(voice_text, items)
        if not category:
            self.publish_error("没有识别到目标大类")
            return

        selected_item = self.select_item_with_spark(category, items)
        if not selected_item:
            selected_item = self.select_item_locally(category, items)
        if not selected_item:
            self.publish_error("没有从二维码中找到目标大类对应货品")
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
        self.publish_state(STATE_DONE)

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

    def select_item_with_spark(self, category, items):
        if not self.spark_ready():
            return ""
        prompt = (
            "你是无人工厂现实环境货品筛选节点。"
            "仿真环境任务全部忽略。"
            "目标大类只能是食品加工类、日用品类、电子产品类。"
            "请从候选货品中选择唯一属于目标大类的货品，并返回严格JSON："
            "{\"selected_item\":\"货品名称\"}。"
            "目标大类：{}。候选货品：{}。"
        ).format(category, "、".join(items))
        data = self.call_spark(prompt, max_tokens=120)
        content = self.extract_spark_content(data)
        obj = first_json_object(content)
        if isinstance(obj, dict):
            selected = normalize_text(obj.get("selected_item"))
            if selected in items:
                return selected
        for item in items:
            if item and item in content:
                return item
        return ""

    def select_item_locally(self, category, items):
        for item in items:
            if LOCAL_ITEM_CATEGORY.get(item) == category:
                return item
        for item in items:
            item_category = self.guess_item_category_by_keyword(item)
            if item_category == category:
                return item
        return ""

    def guess_item_category_by_keyword(self, item):
        text = normalize_text(item)
        if re.search(r"米|肉|果|奶|蛋|面包|食品|香蕉|苹果|梨|饼干|蔬菜", text):
            return "食品加工类"
        if re.search(r"毛巾|牙|皂|纸|棉|衣|杯|伞|日用|洗发", text):
            return "日用品类"
        if re.search(r"电|机|芯片|手机|电脑|鼠标|键盘|耳机|相机|平板|充电", text):
            return "电子产品类"
        return ""

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
            resp = requests.post(
                self.spark_url, headers=headers, json=payload,
                timeout=self.spark_timeout_s)
            if resp.status_code != 200:
                rospy.logwarn("Spark request failed: %s %s", resp.status_code, resp.text[:200])
                return None
            return resp.json()
        except Exception as exc:
            rospy.logwarn("Spark request exception: %s", exc)
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
