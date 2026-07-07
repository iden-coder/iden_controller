#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import ast
import math
import os
import sys

try:
    import yaml
except Exception:
    yaml = None


DEFAULT_MAP_YAML = "/home/ucar/instant_ws/src/ucar_nav/maps/6.3.1.yaml"
DEFAULT_WAYPOINTS_YAML = (
    "/home/ucar/instant_ws/src/iden_controller/config/"
    "navfn_success1_guarded_waypoints.yaml"
)


def yaw_from_pose7(pose7):
    x, y, z, w = pose7[3], pose7[4], pose7[5], pose7[6]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def load_yaml_file(path):
    if yaml is None:
        return load_simple_yaml_file(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_scalar(text):
    text = text.strip()
    if text == "":
        return ""
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except Exception:
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [parse_scalar(part.strip()) for part in inner.split(",")]
    if text.startswith("{") and text.endswith("}"):
        return ast.literal_eval(text)
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except Exception:
        return text.strip("\"'")


def load_simple_yaml_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.split("#", 1)[0].rstrip() for ln in f if ln.split("#", 1)[0].strip()]
    root = {}
    stack = [(-1, root)]
    for idx, line in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if text.startswith("- "):
            parent.append(parse_scalar(text[2:].strip()))
            continue
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
            continue
        next_is_list = False
        for future in lines[idx + 1:]:
            future_indent = len(future) - len(future.lstrip(" "))
            if future_indent <= indent:
                break
            next_is_list = future.strip().startswith("- ")
            break
        container = [] if next_is_list else {}
        parent[key] = container
        stack.append((indent, container))
    return root


def load_waypoints(path):
    data = load_yaml_file(path)
    return {
        "nav_points": data.get("nav_points", {}),
        "patrol_path": data.get("patrol_path", []),
        "force_moves": data.get("force_moves", {}),
        "narrow_points": data.get("narrow_points", []),
    }


def read_pgm(path):
    with open(path, "rb") as f:
        def read_token():
            token = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    return None
                if ch == b"#":
                    f.readline()
                    continue
                if ch.isspace():
                    continue
                token.extend(ch)
                break
            while True:
                ch = f.read(1)
                if not ch or ch.isspace():
                    break
                if ch == b"#":
                    f.readline()
                    break
                token.extend(ch)
            return bytes(token)

        magic = read_token()
        if magic not in (b"P5", b"P2"):
            raise RuntimeError("unsupported PGM type: %r" % (magic,))
        width = int(read_token())
        height = int(read_token())
        maxval = int(read_token())
        if maxval <= 0 or maxval > 255:
            raise RuntimeError("unsupported PGM maxval: %s" % maxval)

        if magic == b"P5":
            data = bytearray(f.read(width * height))
            if len(data) != width * height:
                raise RuntimeError("PGM data is shorter than expected")
            return width, height, data

        values = []
        for _ in range(width * height):
            tok = read_token()
            if tok is None:
                raise RuntimeError("PGM ASCII data is shorter than expected")
            values.append(int(tok))
        return width, height, bytearray(values)


class StaticMap:
    def __init__(self, map_yaml, unknown_is_obstacle=False):
        self.map_yaml = os.path.abspath(map_yaml)
        meta = load_yaml_file(self.map_yaml)
        image = meta["image"]
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(self.map_yaml), image)
        self.image_path = image
        self.resolution = float(meta["resolution"])
        self.origin = [float(v) for v in meta.get("origin", [0.0, 0.0, 0.0])]
        self.negate = int(meta.get("negate", 0))
        self.occupied_thresh = float(meta.get("occupied_thresh", 0.65))
        self.free_thresh = float(meta.get("free_thresh", 0.196))
        self.unknown_is_obstacle = bool(unknown_is_obstacle)

        self.width, self.height, pixels = read_pgm(self.image_path)
        self.state = bytearray(self.width * self.height)
        self.blocked = bytearray(self.width * self.height)

        for idx, value in enumerate(pixels):
            if self.negate:
                occ = float(value) / 255.0
            else:
                occ = float(255 - value) / 255.0
            if occ >= self.occupied_thresh:
                self.state[idx] = 2
                self.blocked[idx] = 1
            elif occ <= self.free_thresh:
                self.state[idx] = 0
            else:
                self.state[idx] = 1
                if self.unknown_is_obstacle:
                    self.blocked[idx] = 1

        self.dist_cells = self._distance_transform()

    def _distance_transform(self):
        inf = 10 ** 9
        dist = [inf] * (self.width * self.height)
        q = collections.deque()
        for idx, blocked in enumerate(self.blocked):
            if blocked:
                dist[idx] = 0
                q.append(idx)
        while q:
            idx = q.popleft()
            x = idx % self.width
            y = idx // self.width
            nd = dist[idx] + 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        ni = ny * self.width + nx
                        if dist[ni] > nd:
                            dist[ni] = nd
                            q.append(ni)
        return dist

    def world_to_cell(self, x, y):
        ox, oy, theta = self.origin[0], self.origin[1], self.origin[2]
        dx = x - ox
        dy = y - oy
        c = math.cos(-theta)
        s = math.sin(-theta)
        mx_f = (c * dx - s * dy) / self.resolution
        my_f = (s * dx + c * dy) / self.resolution
        mx = int(math.floor(mx_f))
        my = int(math.floor(my_f))
        row = self.height - 1 - my
        return mx, my, row

    def clearance(self, x, y):
        mx, my, row = self.world_to_cell(x, y)
        if not (0 <= mx < self.width and 0 <= row < self.height):
            return None, "OUT"
        idx = row * self.width + mx
        if self.state[idx] == 2:
            return 0.0, "OCC"
        if self.state[idx] == 1:
            return self.dist_cells[idx] * self.resolution, "UNKNOWN"
        dist = self.dist_cells[idx]
        if dist >= 10 ** 9:
            return float("inf"), "FREE"
        return dist * self.resolution, "FREE"


def sample_segment_clearance(static_map, a, b, step=0.03):
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    count = max(2, int(math.ceil(length / max(step, 1e-3))))
    best = float("inf")
    best_state = "FREE"
    best_xy = (ax, ay)
    for i in range(count + 1):
        t = float(i) / float(count)
        x = ax + (bx - ax) * t
        y = ay + (by - ay) * t
        clearance, state = static_map.clearance(x, y)
        if clearance is None:
            return None, "OUT", (x, y)
        if clearance < best:
            best = clearance
            best_state = state
            best_xy = (x, y)
    return best, best_state, best_xy


def sample_force_move_clearance(static_map, pose7, move, step=0.03):
    distance = abs(float(move.get("distance", 0.0)))
    direction = 1.0 if float(move.get("direction", 1.0)) >= 0.0 else -1.0
    yaw = yaw_from_pose7(pose7)
    x0, y0 = pose7[0], pose7[1]
    count = max(2, int(math.ceil(distance / max(step, 1e-3))))
    best = float("inf")
    best_state = "FREE"
    best_xy = (x0, y0)
    for i in range(count + 1):
        d = direction * distance * float(i) / float(count)
        x = x0 + math.cos(yaw) * d
        y = y0 + math.sin(yaw) * d
        clearance, state = static_map.clearance(x, y)
        if clearance is None:
            return None, "OUT", (x, y)
        if clearance < best:
            best = clearance
            best_state = state
            best_xy = (x, y)
    return best, best_state, best_xy


def run_preflight(map_yaml, nav_points, patrol_path, force_moves, options=None, emit=print):
    opts = {
        "fail_goal_clearance": 0.13,
        "warn_goal_clearance": 0.20,
        "fail_force_clearance": 0.18,
        "warn_segment_clearance": 0.14,
        "strict_segment_fail": False,
        "unknown_is_obstacle": False,
    }
    if options:
        opts.update(options)

    static_map = StaticMap(map_yaml, unknown_is_obstacle=opts["unknown_is_obstacle"])
    fail = False
    warn = False
    emit("PRECHECK map=%s image=%s size=%dx%d res=%.3f origin=%s" % (
        static_map.map_yaml, static_map.image_path, static_map.width,
        static_map.height, static_map.resolution, static_map.origin))

    for name in patrol_path:
        if name not in nav_points:
            emit("FAIL point %s missing from nav_points" % name)
            fail = True
            continue
        pose = nav_points[name]
        clearance, state = static_map.clearance(float(pose[0]), float(pose[1]))
        if clearance is None or state in ("OUT", "OCC", "UNKNOWN"):
            emit("FAIL point %s state=%s clearance=%s" % (name, state, clearance))
            fail = True
            continue
        if clearance < opts["fail_goal_clearance"]:
            emit("FAIL point %s clearance=%.3fm below %.3fm" % (
                name, clearance, opts["fail_goal_clearance"]))
            fail = True
        elif clearance < opts["warn_goal_clearance"]:
            emit("WARN point %s clearance=%.3fm below %.3fm" % (
                name, clearance, opts["warn_goal_clearance"]))
            warn = True
        else:
            emit("OK point %s clearance=%.3fm" % (name, clearance))

    for a, b in zip(patrol_path, patrol_path[1:]):
        if a not in nav_points or b not in nav_points:
            continue
        clearance, state, xy = sample_segment_clearance(
            static_map,
            (float(nav_points[a][0]), float(nav_points[a][1])),
            (float(nav_points[b][0]), float(nav_points[b][1])),
        )
        if clearance is None or state == "OUT":
            emit("WARN segment %s->%s leaves map near %.3f %.3f" % (a, b, xy[0], xy[1]))
            warn = True
        elif clearance < opts["warn_segment_clearance"]:
            level = "FAIL" if opts["strict_segment_fail"] else "WARN"
            emit("%s segment %s->%s straight-line clearance=%.3fm near %.3f %.3f" % (
                level, a, b, clearance, xy[0], xy[1]))
            warn = True
            if opts["strict_segment_fail"]:
                fail = True

    for name, move in (force_moves or {}).items():
        if name not in nav_points:
            emit("FAIL force_move %s has no nav point" % name)
            fail = True
            continue
        clearance, state, xy = sample_force_move_clearance(
            static_map, nav_points[name], move,
            step=float(move.get("step_distance", 0.03)))
        if clearance is None or state in ("OUT", "OCC", "UNKNOWN"):
            emit("FAIL force_move %s state=%s near %.3f %.3f" % (name, state, xy[0], xy[1]))
            fail = True
        elif clearance < opts["fail_force_clearance"]:
            emit("FAIL force_move %s clearance=%.3fm below %.3fm near %.3f %.3f" % (
                name, clearance, opts["fail_force_clearance"], xy[0], xy[1]))
            fail = True
        else:
            emit("OK force_move %s clearance=%.3fm" % (name, clearance))

    if fail:
        emit("PRECHECK RESULT: FAIL")
        return False
    if warn:
        emit("PRECHECK RESULT: PASS_WITH_WARNINGS")
    else:
        emit("PRECHECK RESULT: PASS")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-yaml", default=DEFAULT_MAP_YAML)
    parser.add_argument("--waypoints-yaml", default=DEFAULT_WAYPOINTS_YAML)
    parser.add_argument("--fail-goal-clearance", type=float, default=0.13)
    parser.add_argument("--warn-goal-clearance", type=float, default=0.20)
    parser.add_argument("--fail-force-clearance", type=float, default=0.18)
    parser.add_argument("--warn-segment-clearance", type=float, default=0.14)
    parser.add_argument("--strict-segment-fail", action="store_true")
    parser.add_argument("--unknown-is-obstacle", action="store_true")
    args = parser.parse_args()

    waypoints = load_waypoints(args.waypoints_yaml)
    ok = run_preflight(
        args.map_yaml,
        waypoints["nav_points"],
        waypoints["patrol_path"],
        waypoints["force_moves"],
        {
            "fail_goal_clearance": args.fail_goal_clearance,
            "warn_goal_clearance": args.warn_goal_clearance,
            "fail_force_clearance": args.fail_force_clearance,
            "warn_segment_clearance": args.warn_segment_clearance,
            "strict_segment_fail": args.strict_segment_fail,
            "unknown_is_obstacle": args.unknown_is_obstacle,
        },
    )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
