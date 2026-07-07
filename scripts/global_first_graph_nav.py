#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import heapq
import math
import os
import sys
import time
from collections import defaultdict, deque


INF = 1.0e18


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def norm_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


class GridMap:
    def __init__(self, width, height, resolution, origin, data,
                 occupied_threshold=50, unknown_is_obstacle=True):
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.origin_yaw = float(origin[2]) if len(origin) > 2 else 0.0
        self.data = list(data)
        self.occupied_threshold = int(occupied_threshold)
        self.unknown_is_obstacle = bool(unknown_is_obstacle)
        self.dist_cells = None
        self._cos = math.cos(self.origin_yaw)
        self._sin = math.sin(self.origin_yaw)

    def index(self, mx, my):
        return my * self.width + mx

    def in_bounds(self, mx, my):
        return 0 <= mx < self.width and 0 <= my < self.height

    def world_to_map(self, wx, wy):
        dx = wx - self.origin_x
        dy = wy - self.origin_y
        lx = self._cos * dx + self._sin * dy
        ly = -self._sin * dx + self._cos * dy
        mx = int(math.floor(lx / self.resolution))
        my = int(math.floor(ly / self.resolution))
        if not self.in_bounds(mx, my):
            return None
        return mx, my

    def map_to_world(self, mx, my):
        lx = (mx + 0.5) * self.resolution
        ly = (my + 0.5) * self.resolution
        wx = self.origin_x + self._cos * lx - self._sin * ly
        wy = self.origin_y + self._sin * lx + self._cos * ly
        return wx, wy

    def is_occupied_cell(self, mx, my):
        if not self.in_bounds(mx, my):
            return True
        value = self.data[self.index(mx, my)]
        if value < 0:
            return self.unknown_is_obstacle
        return value >= self.occupied_threshold

    def compute_distance_field(self):
        n = self.width * self.height
        dist = [INF] * n
        for my in range(self.height):
            row = my * self.width
            for mx in range(self.width):
                if self.is_occupied_cell(mx, my):
                    dist[row + mx] = 0.0

        diag = math.sqrt(2.0)
        for my in range(self.height):
            row = my * self.width
            prev = row - self.width
            for mx in range(self.width):
                i = row + mx
                best = dist[i]
                if mx > 0:
                    best = min(best, dist[i - 1] + 1.0)
                if my > 0:
                    best = min(best, dist[prev + mx] + 1.0)
                    if mx > 0:
                        best = min(best, dist[prev + mx - 1] + diag)
                    if mx + 1 < self.width:
                        best = min(best, dist[prev + mx + 1] + diag)
                dist[i] = best

        for my in range(self.height - 1, -1, -1):
            row = my * self.width
            nxt = row + self.width
            for mx in range(self.width - 1, -1, -1):
                i = row + mx
                best = dist[i]
                if mx + 1 < self.width:
                    best = min(best, dist[i + 1] + 1.0)
                if my + 1 < self.height:
                    best = min(best, dist[nxt + mx] + 1.0)
                    if mx > 0:
                        best = min(best, dist[nxt + mx - 1] + diag)
                    if mx + 1 < self.width:
                        best = min(best, dist[nxt + mx + 1] + diag)
                dist[i] = best

        self.dist_cells = dist

    def clearance_m(self, mx, my):
        if not self.in_bounds(mx, my):
            return 0.0
        if self.dist_cells is None:
            self.compute_distance_field()
        d = self.dist_cells[self.index(mx, my)]
        if d >= INF * 0.5:
            return 999.0
        return d * self.resolution

    def is_safe_cell(self, mx, my, clearance_m):
        if self.is_occupied_cell(mx, my):
            return False
        return self.clearance_m(mx, my) >= clearance_m

    def nearest_safe_cell(self, mx, my, clearance_m, max_radius_m):
        if not self.in_bounds(mx, my):
            return None
        if self.is_safe_cell(mx, my, clearance_m):
            return mx, my

        max_radius = max(1, int(math.ceil(max_radius_m / self.resolution)))
        seen = set([(mx, my)])
        q = deque([(mx, my)])
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while q:
            cx, cy = q.popleft()
            dx0 = cx - mx
            dy0 = cy - my
            if dx0 * dx0 + dy0 * dy0 > max_radius * max_radius:
                continue
            if self.is_safe_cell(cx, cy, clearance_m):
                return cx, cy
            for dx, dy in dirs:
                nx = cx + dx
                ny = cy + dy
                if not self.in_bounds(nx, ny) or (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny))
        return None

    def iter_line_cells(self, a, b):
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x = x0
        y = y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                return
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def line_is_safe(self, a, b, clearance_m):
        for x, y in self.iter_line_cells(a, b):
            if not self.is_safe_cell(x, y, clearance_m):
                return False
        return True

    def line_cost(self, a, b, clearance_m, preferred_clearance_m, proximity_weight):
        min_clear = INF
        penalty_sum = 0.0
        count = 0
        for x, y in self.iter_line_cells(a, b):
            if not self.is_safe_cell(x, y, clearance_m):
                return None, 0.0
            clear = self.clearance_m(x, y)
            min_clear = min(min_clear, clear)
            if preferred_clearance_m > clearance_m and clear < preferred_clearance_m:
                t = (preferred_clearance_m - clear) / max(
                    preferred_clearance_m - clearance_m, 1.0e-6)
                penalty_sum += t * t
            count += 1
        length = math.hypot(a[0] - b[0], a[1] - b[1]) * self.resolution
        avg_penalty = penalty_sum / max(count, 1)
        cost = length * (1.0 + proximity_weight * avg_penalty)
        return cost, min_clear


class Roadmap:
    def __init__(self, nodes, clearances, stride_cells, connect_radius_cells):
        self.nodes = nodes
        self.clearances = clearances
        self.stride_cells = stride_cells
        self.bin_size = max(1, connect_radius_cells)
        self.bins = defaultdict(list)
        self.neighbor_cache = {}
        for idx, (mx, my) in enumerate(self.nodes):
            self.bins[(mx // self.bin_size, my // self.bin_size)].append(idx)


class GlobalFirstGraphPlanner:
    def __init__(self, grid, params):
        self.grid = grid
        self.params = dict(params)
        self.roadmaps = {}
        if self.grid.dist_cells is None:
            self.grid.compute_distance_field()

    def plan(self, start_world, goal_world):
        hard = float(self.params.get("hard_clearance_m", 0.16))
        emergency = float(self.params.get("emergency_min_clearance_m", hard))
        allow_fallback = bool(self.params.get("allow_clearance_fallback", True))
        attempts = [hard]
        if allow_fallback and emergency < hard:
            attempts.extend([(hard + emergency) * 0.5, emergency])

        last_error = "not attempted"
        for clearance in attempts:
            result = self._plan_with_clearance(start_world, goal_world, clearance)
            if result["ok"]:
                result["clearance_used_m"] = clearance
                return result
            last_error = result.get("error", "unknown failure")
        return {"ok": False, "error": last_error}

    def _plan_with_clearance(self, start_world, goal_world, clearance_m):
        snap_radius = float(self.params.get("goal_snap_radius_m", 0.45))
        start_cell = self.grid.world_to_map(start_world[0], start_world[1])
        goal_cell = self.grid.world_to_map(goal_world[0], goal_world[1])
        if start_cell is None:
            return {"ok": False, "error": "start outside map"}
        if goal_cell is None:
            return {"ok": False, "error": "goal outside map"}

        raw_start_cell = start_cell
        raw_goal_cell = goal_cell
        start_cell = self.grid.nearest_safe_cell(
            start_cell[0], start_cell[1], clearance_m, snap_radius)
        goal_cell = self.grid.nearest_safe_cell(
            goal_cell[0], goal_cell[1], clearance_m, snap_radius)
        if start_cell is None:
            return {"ok": False, "error": "no safe start near current pose"}
        if goal_cell is None:
            return {"ok": False, "error": "no safe goal near requested goal"}

        graph_error = None
        if bool(self.params.get("graph_enabled", True)):
            graph_result = self._graph_plan(start_cell, goal_cell, clearance_m)
            if graph_result["ok"]:
                return self._finish_result(
                    graph_result["cells"], start_cell, goal_cell,
                    raw_start_cell, raw_goal_cell, clearance_m,
                    graph_result["method"], graph_result.get("roadmap_nodes", 0))
            graph_error = graph_result.get("error", "graph failed")

        if bool(self.params.get("fallback_grid_enabled", True)):
            max_time = float(self.params.get("grid_max_planning_time_s",
                                             self.params.get("max_planning_time_s", 5.0)))
            cells = self._grid_astar(start_cell, goal_cell, clearance_m, max_time)
            if cells:
                return self._finish_result(
                    cells, start_cell, goal_cell, raw_start_cell, raw_goal_cell,
                    clearance_m, "GRID_FALLBACK", 0)
            if graph_error:
                return {"ok": False, "error": graph_error + "; grid fallback failed"}
            return {"ok": False, "error": "grid fallback failed"}

        return {"ok": False, "error": graph_error or "graph disabled"}

    def _finish_result(self, cells, start_cell, goal_cell, raw_start_cell,
                       raw_goal_cell, clearance_m, method, roadmap_nodes):
        smooth_clearance = max(
            clearance_m,
            float(self.params.get("smooth_clearance_m", clearance_m)))
        smooth_cells = self._smooth(cells, smooth_clearance)
        path_world = self._densify(smooth_cells)
        min_clearance = self._path_min_clearance(cells)
        active_goal = self.grid.map_to_world(goal_cell[0], goal_cell[1])
        return {
            "ok": True,
            "cells": cells,
            "smooth_cells": smooth_cells,
            "path_world": path_world,
            "start_cell": start_cell,
            "goal_cell": goal_cell,
            "raw_start_cell": raw_start_cell,
            "raw_goal_cell": raw_goal_cell,
            "active_goal_world": active_goal,
            "min_clearance_m": min_clearance,
            "path_length_m": self._path_length(path_world),
            "planner_method": method,
            "roadmap_nodes": roadmap_nodes,
        }

    def _build_roadmap(self, clearance_m):
        key = round(clearance_m, 3)
        if key in self.roadmaps:
            return self.roadmaps[key]

        stride_m = float(self.params.get("graph_sample_stride_m", 0.16))
        stride_cells = max(2, int(math.ceil(stride_m / self.grid.resolution)))
        connect_radius_m = float(self.params.get("graph_connect_radius_m", 0.75))
        connect_radius_cells = max(1, int(math.ceil(connect_radius_m / self.grid.resolution)))
        max_nodes = int(self.params.get("graph_max_nodes", 4200))

        nodes = []
        clearances = []
        while True:
            nodes = []
            clearances = []
            seen = set()
            for by in range(0, self.grid.height, stride_cells):
                for bx in range(0, self.grid.width, stride_cells):
                    best = None
                    best_clear = -1.0
                    y_end = min(self.grid.height, by + stride_cells)
                    x_end = min(self.grid.width, bx + stride_cells)
                    for my in range(by, y_end):
                        for mx in range(bx, x_end):
                            if not self.grid.is_safe_cell(mx, my, clearance_m):
                                continue
                            clear = self.grid.clearance_m(mx, my)
                            if clear > best_clear:
                                best = (mx, my)
                                best_clear = clear
                    if best is not None and best not in seen:
                        seen.add(best)
                        nodes.append(best)
                        clearances.append(best_clear)
            if len(nodes) <= max_nodes or stride_cells >= 24:
                break
            stride_cells += max(1, stride_cells // 3)

        roadmap = Roadmap(nodes, clearances, stride_cells, connect_radius_cells)
        self.roadmaps[key] = roadmap
        return roadmap

    def _graph_plan(self, start_cell, goal_cell, clearance_m):
        preferred = float(self.params.get("preferred_clearance_m", clearance_m))
        prox_weight = float(self.params.get("proximity_weight", 2.4))
        graph_time = float(self.params.get("graph_max_planning_time_s",
                                           self.params.get("max_planning_time_s", 5.0)))
        connect_radius_m = float(self.params.get("graph_connect_radius_m", 0.75))
        start_goal_radius_m = float(self.params.get("graph_start_goal_radius_m", 0.95))
        heuristic_weight = float(self.params.get("heuristic_weight", 1.05))
        max_neighbors = int(self.params.get("graph_max_neighbors", 18))
        start_goal_neighbors = int(self.params.get("graph_start_goal_neighbors", 32))
        start_time = time.time()

        direct_cost, _ = self.grid.line_cost(
            start_cell, goal_cell, clearance_m, preferred, prox_weight)
        if direct_cost is not None:
            return {"ok": True, "cells": [start_cell, goal_cell],
                    "method": "GRAPH_DIRECT", "roadmap_nodes": 0}

        roadmap = self._build_roadmap(clearance_m)
        if not roadmap.nodes:
            return {"ok": False, "error": "graph has no safe nodes"}

        base_n = len(roadmap.nodes)
        start_idx = base_n
        goal_idx = base_n + 1

        def cell_of(idx):
            if idx == start_idx:
                return start_cell
            if idx == goal_idx:
                return goal_cell
            return roadmap.nodes[idx]

        def heuristic(idx):
            c = cell_of(idx)
            return math.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1]) * self.grid.resolution

        def edge_cost(a, b):
            cost, _ = self.grid.line_cost(a, b, clearance_m, preferred, prox_weight)
            return cost

        def static_candidates(cell, radius_m, limit):
            radius_cells = max(1, int(math.ceil(radius_m / self.grid.resolution)))
            r2 = radius_cells * radius_cells
            bx0 = (cell[0] - radius_cells) // roadmap.bin_size
            bx1 = (cell[0] + radius_cells) // roadmap.bin_size
            by0 = (cell[1] - radius_cells) // roadmap.bin_size
            by1 = (cell[1] + radius_cells) // roadmap.bin_size
            candidates = []
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    for idx in roadmap.bins.get((bx, by), []):
                        n = roadmap.nodes[idx]
                        dx = n[0] - cell[0]
                        dy = n[1] - cell[1]
                        d2 = dx * dx + dy * dy
                        if 0 < d2 <= r2:
                            candidates.append((d2, idx))
            candidates.sort()
            return [idx for _, idx in candidates[:max(limit, 1) * 4]]

        def static_neighbors(idx):
            if idx in roadmap.neighbor_cache:
                return roadmap.neighbor_cache[idx]
            src = roadmap.nodes[idx]
            out = []
            for other in static_candidates(src, connect_radius_m, max_neighbors):
                if other == idx:
                    continue
                cost = edge_cost(src, roadmap.nodes[other])
                if cost is None:
                    continue
                out.append((other, cost))
                if len(out) >= max_neighbors:
                    break
            roadmap.neighbor_cache[idx] = out
            return out

        def dynamic_to_static(cell):
            out = []
            for other in static_candidates(cell, start_goal_radius_m, start_goal_neighbors):
                cost = edge_cost(cell, roadmap.nodes[other])
                if cost is None:
                    continue
                out.append((other, cost))
                if len(out) >= start_goal_neighbors:
                    break
            return out

        def neighbors(idx):
            if idx == goal_idx:
                return []
            src = cell_of(idx)
            out = []
            if idx == start_idx:
                out.extend(dynamic_to_static(start_cell))
                cost = edge_cost(start_cell, goal_cell)
                if cost is not None:
                    out.append((goal_idx, cost))
                return out
            out.extend(static_neighbors(idx))
            if math.hypot(src[0] - goal_cell[0], src[1] - goal_cell[1]) * self.grid.resolution <= start_goal_radius_m:
                cost = edge_cost(src, goal_cell)
                if cost is not None:
                    out.append((goal_idx, cost))
            return out

        open_heap = [(heuristic_weight * heuristic(start_idx), 0, start_idx)]
        g_score = {start_idx: 0.0}
        parent = {}
        closed = set()
        pushes = 1

        while open_heap:
            if time.time() - start_time > graph_time:
                return {"ok": False, "error": "graph A* timed out",
                        "roadmap_nodes": base_n}
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_idx:
                cells = []
                idx = current
                while True:
                    cells.append(cell_of(idx))
                    if idx == start_idx:
                        break
                    idx = parent[idx]
                cells.reverse()
                return {"ok": True, "cells": cells, "method": "GRAPH_ROADMAP",
                        "roadmap_nodes": base_n}
            closed.add(current)

            for nxt, cost in neighbors(current):
                if nxt in closed:
                    continue
                tentative = g_score[current] + cost
                if tentative + 1.0e-9 < g_score.get(nxt, INF):
                    parent[nxt] = current
                    g_score[nxt] = tentative
                    pushes += 1
                    f = tentative + heuristic_weight * heuristic(nxt)
                    heapq.heappush(open_heap, (f, pushes, nxt))

        return {"ok": False, "error": "graph A* could not connect start to goal",
                "roadmap_nodes": base_n}

    def _grid_astar(self, start, goal, clearance_m, max_time):
        w = self.grid.width
        h = self.grid.height
        n = w * h
        start_i = self.grid.index(start[0], start[1])
        goal_i = self.grid.index(goal[0], goal[1])
        g_score = [INF] * n
        parent = [-1] * n
        closed = bytearray(n)
        dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
                (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]

        preferred = float(self.params.get("preferred_clearance_m", clearance_m))
        prox_weight = float(self.params.get("proximity_weight", 2.4))
        unknown_cost = float(self.params.get("unknown_cost", 6.0))
        heuristic_weight = float(self.params.get("heuristic_weight", 1.05))
        start_time = time.time()

        def heuristic(mx, my):
            return math.hypot(mx - goal[0], my - goal[1])

        g_score[start_i] = 0.0
        heap = [(heuristic_weight * heuristic(start[0], start[1]), 0, start_i)]
        pushes = 1

        while heap:
            if time.time() - start_time > max_time:
                return None
            _, _, current = heapq.heappop(heap)
            if closed[current]:
                continue
            if current == goal_i:
                return self._reconstruct(parent, current)
            closed[current] = 1
            cx = current % w
            cy = current // w

            for dx, dy, step in dirs:
                nx = cx + dx
                ny = cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                ni = ny * w + nx
                if closed[ni]:
                    continue
                if not self.grid.is_safe_cell(nx, ny, clearance_m):
                    continue
                if dx != 0 and dy != 0:
                    if (not self.grid.is_safe_cell(cx + dx, cy, clearance_m) or
                            not self.grid.is_safe_cell(cx, cy + dy, clearance_m)):
                        continue

                clear = self.grid.clearance_m(nx, ny)
                if preferred > clearance_m and clear < preferred:
                    t = (preferred - clear) / max(preferred - clearance_m, 1.0e-6)
                    proximity = prox_weight * t * t
                else:
                    proximity = 0.0
                extra_unknown = unknown_cost if self.grid.data[ni] < 0 else 0.0
                tentative = g_score[current] + step * (1.0 + proximity + extra_unknown)
                if tentative + 1.0e-9 < g_score[ni]:
                    g_score[ni] = tentative
                    parent[ni] = current
                    pushes += 1
                    f = tentative + heuristic_weight * heuristic(nx, ny)
                    heapq.heappush(heap, (f, pushes, ni))
        return None

    def _reconstruct(self, parent, current):
        w = self.grid.width
        out = []
        while current >= 0:
            out.append((current % w, current // w))
            current = parent[current]
        out.reverse()
        return out

    def _smooth(self, cells, clearance_m):
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1:
                if self.grid.line_is_safe(cells[i], cells[j], clearance_m):
                    break
                j -= 1
            out.append(cells[j])
            i = j
        return out

    def _densify(self, cells):
        spacing = float(self.params.get("path_spacing_m", 0.05))
        if not cells:
            return []
        points = [self.grid.map_to_world(cells[0][0], cells[0][1])]
        for i in range(1, len(cells)):
            ax, ay = self.grid.map_to_world(cells[i - 1][0], cells[i - 1][1])
            bx, by = self.grid.map_to_world(cells[i][0], cells[i][1])
            dist = math.hypot(bx - ax, by - ay)
            steps = max(1, int(math.ceil(dist / max(spacing, 1.0e-3))))
            for k in range(1, steps + 1):
                t = float(k) / float(steps)
                points.append((ax + (bx - ax) * t, ay + (by - ay) * t))
        return points

    def _path_length(self, points):
        total = 0.0
        for i in range(1, len(points)):
            total += math.hypot(points[i][0] - points[i - 1][0],
                                points[i][1] - points[i - 1][1])
        return total

    def _path_min_clearance(self, cells):
        best = INF
        if not cells:
            return 0.0
        for i in range(1, len(cells)):
            for mx, my in self.grid.iter_line_cells(cells[i - 1], cells[i]):
                best = min(best, self.grid.clearance_m(mx, my))
        if best >= INF * 0.5:
            return self.grid.clearance_m(cells[0][0], cells[0][1])
        return best


def parse_simple_yaml(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                parts = [p.strip() for p in value[1:-1].split(",") if p.strip()]
                data[key] = [float(p) for p in parts]
            elif value.startswith('"') and value.endswith('"'):
                data[key] = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                data[key] = value[1:-1]
            else:
                try:
                    if "." in value or "e" in value.lower():
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value
    return data


def read_pgm(path):
    with open(path, "rb") as f:
        def token():
            buf = bytearray()
            while True:
                c = f.read(1)
                if not c:
                    return None
                if c == b"#":
                    f.readline()
                    continue
                if c.isspace():
                    if buf:
                        return bytes(buf)
                    continue
                buf.extend(c)

        magic = token()
        if magic not in (b"P5", b"P2"):
            raise ValueError("unsupported PGM magic %r" % (magic,))
        width = int(token())
        height = int(token())
        maxval = int(token())
        if maxval <= 0 or maxval > 255:
            raise ValueError("unsupported PGM max value %d" % maxval)
        if magic == b"P5":
            raw = f.read(width * height)
            if len(raw) != width * height:
                raise ValueError("PGM data is truncated")
            pixels = list(raw)
        else:
            pixels = [int(token()) for _ in range(width * height)]
    return width, height, pixels


def load_map_yaml(path, occupied_threshold=50, unknown_is_obstacle=True):
    meta = parse_simple_yaml(path)
    image = str(meta["image"])
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(path), image)
    width, height, pixels = read_pgm(image)
    resolution = float(meta["resolution"])
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    negate = int(meta.get("negate", 0))
    occ_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.196))

    data = [0] * (width * height)
    for img_y in range(height):
        map_y = height - 1 - img_y
        for x in range(width):
            p = pixels[img_y * width + x]
            occ = (float(p) / 255.0) if negate else (255.0 - float(p)) / 255.0
            if occ > occ_thresh:
                value = 100
            elif occ < free_thresh:
                value = 0
            else:
                value = -1
            data[map_y * width + x] = value
    return GridMap(width, height, resolution, origin, data,
                   occupied_threshold, unknown_is_obstacle)


def default_params_from_args(args):
    return {
        "hard_clearance_m": args.hard_clearance,
        "preferred_clearance_m": args.preferred_clearance,
        "emergency_min_clearance_m": args.emergency_clearance,
        "smooth_clearance_m": args.hard_clearance,
        "allow_clearance_fallback": True,
        "goal_snap_radius_m": 0.45,
        "max_planning_time_s": 8.0,
        "grid_max_planning_time_s": 5.0,
        "graph_enabled": True,
        "fallback_grid_enabled": True,
        "graph_sample_stride_m": args.graph_stride,
        "graph_connect_radius_m": args.graph_connect_radius,
        "graph_start_goal_radius_m": args.graph_start_goal_radius,
        "graph_max_neighbors": 18,
        "graph_start_goal_neighbors": 32,
        "graph_max_nodes": 4200,
        "graph_max_planning_time_s": 4.0,
        "heuristic_weight": 1.05,
        "proximity_weight": 2.6,
        "unknown_cost": 6.0,
        "path_spacing_m": 0.05,
    }


def offline_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--start", required=True, help="x,y")
    parser.add_argument("--goal", required=True, help="x,y")
    parser.add_argument("--hard-clearance", type=float, default=0.16)
    parser.add_argument("--preferred-clearance", type=float, default=0.30)
    parser.add_argument("--emergency-clearance", type=float, default=0.10)
    parser.add_argument("--graph-stride", type=float, default=0.16)
    parser.add_argument("--graph-connect-radius", type=float, default=0.78)
    parser.add_argument("--graph-start-goal-radius", type=float, default=1.00)
    args = parser.parse_args(argv)

    def xy(text):
        a, b = text.split(",", 1)
        return float(a), float(b)

    grid = load_map_yaml(args.map_yaml)
    t0 = time.time()
    grid.compute_distance_field()
    planner = GlobalFirstGraphPlanner(grid, default_params_from_args(args))
    result = planner.plan(xy(args.start), xy(args.goal))
    elapsed = time.time() - t0
    if not result["ok"]:
        print("PLAN_FAIL: %s" % result["error"])
        return 2
    print("PLAN_OK")
    print("planner_method=%s" % result["planner_method"])
    print("roadmap_nodes=%d" % result["roadmap_nodes"])
    print("clearance_used_m=%.3f" % result["clearance_used_m"])
    print("min_clearance_m=%.3f" % result["min_clearance_m"])
    print("path_length_m=%.3f" % result["path_length_m"])
    print("path_points=%d smooth_points=%d raw_cells=%d" % (
        len(result["path_world"]),
        len(result["smooth_cells"]),
        len(result["cells"])))
    print("active_goal_world=%.3f,%.3f" % result["active_goal_world"])
    print("elapsed_s=%.3f" % elapsed)
    return 0


class RosGlobalFirstGraphNavigator:
    def __init__(self):
        import rospy
        import tf
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Path
        from std_msgs.msg import String

        self.rospy = rospy
        self.tf = tf
        self.Twist = Twist
        self.Path = Path
        self.String = String

        rospy.init_node("global_first_graph_nav")
        self.tf_listener = tf.TransformListener()

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.goal_x = rospy.get_param("~goal_x", -1.50)
        self.goal_y = rospy.get_param("~goal_y", -0.40)
        self.goal_yaw = rospy.get_param("~goal_yaw", math.pi)
        self.require_goal_yaw = rospy.get_param("~require_goal_yaw", False)
        self.goal_tolerance = rospy.get_param("~goal_tolerance_m", 0.12)
        self.goal_yaw_tolerance = math.radians(
            rospy.get_param("~goal_yaw_tolerance_deg", 12.0))
        self.goal_hold_s = rospy.get_param("~goal_hold_s", 0.8)

        self.publish_initial_pose_on_start = rospy.get_param(
            "~publish_initial_pose_on_start", True)
        self.initial_pose_x = rospy.get_param("~initial_pose_x", 1.07)
        self.initial_pose_y = rospy.get_param("~initial_pose_y", 0.0)
        self.initial_pose_yaw = rospy.get_param("~initial_pose_yaw", 0.0)
        self.initial_pose_repeat_s = rospy.get_param("~initial_pose_repeat_s", 3.0)
        self.initial_pose_period_s = rospy.get_param("~initial_pose_period_s", 0.25)
        self.require_start_near_initial = rospy.get_param(
            "~require_start_near_initial", True)
        self.start_pose_tolerance_m = rospy.get_param("~start_pose_tolerance_m", 0.45)
        self.start_yaw_tolerance_deg = rospy.get_param("~start_yaw_tolerance_deg", 80.0)

        self.params = {
            "hard_clearance_m": rospy.get_param("~hard_clearance_m", 0.16),
            "preferred_clearance_m": rospy.get_param("~preferred_clearance_m", 0.30),
            "emergency_min_clearance_m": rospy.get_param(
                "~emergency_min_clearance_m", 0.10),
            "smooth_clearance_m": rospy.get_param("~smooth_clearance_m", 0.16),
            "allow_clearance_fallback": rospy.get_param(
                "~allow_clearance_fallback", True),
            "goal_snap_radius_m": rospy.get_param("~goal_snap_radius_m", 0.45),
            "max_planning_time_s": rospy.get_param("~max_planning_time_s", 6.0),
            "grid_max_planning_time_s": rospy.get_param(
                "~grid_max_planning_time_s", 4.0),
            "graph_enabled": rospy.get_param("~graph_enabled", True),
            "fallback_grid_enabled": rospy.get_param("~fallback_grid_enabled", True),
            "graph_sample_stride_m": rospy.get_param("~graph_sample_stride_m", 0.16),
            "graph_connect_radius_m": rospy.get_param("~graph_connect_radius_m", 0.78),
            "graph_start_goal_radius_m": rospy.get_param(
                "~graph_start_goal_radius_m", 1.00),
            "graph_max_neighbors": rospy.get_param("~graph_max_neighbors", 18),
            "graph_start_goal_neighbors": rospy.get_param(
                "~graph_start_goal_neighbors", 32),
            "graph_max_nodes": rospy.get_param("~graph_max_nodes", 4200),
            "graph_max_planning_time_s": rospy.get_param(
                "~graph_max_planning_time_s", 3.5),
            "proximity_weight": rospy.get_param("~proximity_weight", 2.6),
            "unknown_cost": rospy.get_param("~unknown_cost", 6.0),
            "heuristic_weight": rospy.get_param("~heuristic_weight", 1.05),
            "path_spacing_m": rospy.get_param("~path_spacing_m", 0.05),
        }
        self.occupied_threshold = rospy.get_param("~occupied_threshold", 50)
        self.unknown_is_obstacle = rospy.get_param("~unknown_is_obstacle", True)

        self.control_rate_hz = rospy.get_param("~control_rate_hz", 12.0)
        self.lookahead_dist = rospy.get_param("~lookahead_dist_m", 0.30)
        self.rotate_in_place_angle = math.radians(
            rospy.get_param("~rotate_in_place_angle_deg", 58.0))
        self.max_linear = rospy.get_param("~max_linear_vel", 0.22)
        self.min_tracking_speed = rospy.get_param("~min_tracking_speed", 0.040)
        self.max_angular = rospy.get_param("~max_angular_vel", 0.72)
        self.turn_creep_speed = rospy.get_param("~turn_creep_speed", 0.026)
        self.turn_creep_front_min = rospy.get_param("~turn_creep_front_min", 0.30)
        self.turn_creep_side_min = rospy.get_param("~turn_creep_side_min", 0.20)
        self.k_heading = rospy.get_param("~k_heading", 1.30)
        self.k_lateral = rospy.get_param("~k_lateral", 0.58)
        self.linear_accel = rospy.get_param("~linear_accel", 0.18)
        self.angular_accel = rospy.get_param("~angular_accel", 1.00)

        self.front_angle_deg = rospy.get_param("~front_angle_deg", 36.0)
        self.side_angle_min_deg = rospy.get_param("~side_angle_min_deg", 35.0)
        self.side_angle_max_deg = rospy.get_param("~side_angle_max_deg", 80.0)
        self.front_stop_m = rospy.get_param("~front_stop_m", 0.22)
        self.front_slow_m = rospy.get_param("~front_slow_m", 0.58)
        self.side_stop_m = rospy.get_param("~side_stop_m", 0.16)
        self.side_slow_m = rospy.get_param("~side_slow_m", 0.32)
        self.scan_timeout_s = rospy.get_param("~scan_timeout_s", 0.70)
        self.pose_timeout_s = rospy.get_param("~pose_timeout_s", 5.0)
        self.front_turn_speed = rospy.get_param("~front_turn_speed", 0.18)
        self.front_replan_after_s = rospy.get_param("~front_replan_after_s", 3.0)

        self.replan_min_interval_s = rospy.get_param("~replan_min_interval_s", 2.0)
        self.blocked_replan_s = rospy.get_param("~blocked_replan_s", 1.5)
        self.stuck_timeout_s = rospy.get_param("~stuck_timeout_s", 8.0)
        self.progress_dist_m = rospy.get_param("~progress_dist_m", 0.06)

        self.grid = None
        self.planner = None
        self.pose = None
        self.pose_time = rospy.Time(0)
        self.scan = None
        self.scan_time = rospy.Time(0)
        self.front = float("inf")
        self.left = float("inf")
        self.right = float("inf")
        self.rear = float("inf")

        self.path_world = []
        self.path_index = 0
        self.active_goal = (self.goal_x, self.goal_y)
        self.finished = False
        self.start_pose_accepted = not self.require_start_near_initial
        self.initial_pose_start_time = None
        self.initial_pose_last_pub = rospy.Time(0)
        self.goal_enter_time = None
        self.last_plan_time = rospy.Time(0)
        self.last_blocked_replan = rospy.Time(0)
        self.last_progress_time = rospy.Time.now()
        self.last_progress_pose = None
        self.last_progress_yaw = None
        self.last_goal_dist = INF
        self.last_cmd = (0.0, 0.0)
        self.last_control_time = rospy.Time.now()
        self.replan_fail_count = 0
        self.front_block_start = None

        self.cmd_pub = rospy.Publisher(
            rospy.get_param("~cmd_vel_topic", "/cmd_vel_raw"),
            Twist, queue_size=1)
        self.path_pub = rospy.Publisher("~path", Path, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("~status", String, queue_size=3, latch=True)
        self.initial_pose_pub = rospy.Publisher(
            rospy.get_param("~initial_pose_topic", "/initialpose"),
            PoseWithCovarianceStamped, queue_size=1)

        from sensor_msgs.msg import LaserScan
        rospy.Subscriber(rospy.get_param("~pose_topic", "/amcl_pose"),
                         PoseWithCovarianceStamped, self.cb_pose, queue_size=1)
        rospy.Subscriber(rospy.get_param("~scan_topic", "/scan"),
                         LaserScan, self.cb_scan, queue_size=1)

        self.load_static_map()
        timer_period = 1.0 / max(self.control_rate_hz, 1.0)
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.control_loop)
        if self.publish_initial_pose_on_start:
            self.initial_pose_start_time = rospy.Time.now()
            self.publish_initial_pose(force=True)
        self.log_status("global_first_graph_nav started; waiting for pose/scan")

    def load_static_map(self):
        import rospy
        from nav_msgs.srv import GetMap
        from nav_msgs.msg import OccupancyGrid

        map_service = rospy.get_param("~map_service", "/static_map")
        map_topic = rospy.get_param("~map_topic", "/map")
        map_yaml = rospy.get_param("~map_yaml", "")
        msg = None

        try:
            rospy.wait_for_service(map_service, timeout=5.0)
            srv = rospy.ServiceProxy(map_service, GetMap)
            msg = srv().map
            rospy.logwarn("GlobalFirstGraphNav: loaded static map from %s", map_service)
        except Exception as exc:
            rospy.logwarn("GlobalFirstGraphNav: static map service failed: %s", str(exc))

        if msg is None:
            try:
                msg = rospy.wait_for_message(map_topic, OccupancyGrid, timeout=5.0)
                rospy.logwarn("GlobalFirstGraphNav: loaded static map from %s", map_topic)
            except Exception as exc:
                rospy.logwarn("GlobalFirstGraphNav: map topic failed: %s", str(exc))

        if msg is not None:
            yaw = yaw_from_quat(msg.info.origin.orientation)
            origin = (msg.info.origin.position.x, msg.info.origin.position.y, yaw)
            self.grid = GridMap(msg.info.width, msg.info.height,
                                msg.info.resolution, origin, msg.data,
                                self.occupied_threshold, self.unknown_is_obstacle)
        elif map_yaml:
            self.grid = load_map_yaml(map_yaml, self.occupied_threshold,
                                      self.unknown_is_obstacle)
            rospy.logwarn("GlobalFirstGraphNav: loaded map yaml fallback %s", map_yaml)
        else:
            raise RuntimeError("no static map available")

        t0 = time.time()
        self.grid.compute_distance_field()
        self.planner = GlobalFirstGraphPlanner(self.grid, self.params)
        rospy.logwarn("GlobalFirstGraphNav: distance field ready in %.2fs",
                      time.time() - t0)

    def cb_pose(self, msg):
        q = msg.pose.pose.orientation
        self.pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quat(q))
        self.pose_time = self.rospy.Time.now()

    def cb_scan(self, msg):
        self.scan = msg
        self.scan_time = self.rospy.Time.now()
        half = self.front_angle_deg
        self.front = self.scan_min(msg, -half, half)
        self.left = self.scan_min(msg, self.side_angle_min_deg,
                                  self.side_angle_max_deg)
        self.right = self.scan_min(msg, -self.side_angle_max_deg,
                                   -self.side_angle_min_deg)
        rear_a = self.scan_min(msg, 145.0, 180.0)
        rear_b = self.scan_min(msg, -180.0, -145.0)
        self.rear = min(rear_a, rear_b)

    def scan_min(self, msg, lo_deg, hi_deg):
        lo = math.radians(lo_deg)
        hi = math.radians(hi_deg)
        best = float("inf")
        for i, value in enumerate(msg.ranges):
            if math.isnan(value) or math.isinf(value):
                continue
            if value < msg.range_min or value > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if lo <= angle <= hi and value < best:
                best = value
        return best

    def scan_fresh(self):
        if self.scan is None:
            return False
        return (self.rospy.Time.now() - self.scan_time).to_sec() <= self.scan_timeout_s

    def pose_fresh(self):
        self.update_pose_from_tf()
        if self.pose is None:
            return False
        if self.pose_timeout_s <= 0.0:
            return True
        return (self.rospy.Time.now() - self.pose_time).to_sec() <= self.pose_timeout_s

    def update_pose_from_tf(self):
        try:
            trans, rot = self.tf_listener.lookupTransform(
                self.map_frame, self.base_frame, self.rospy.Time(0))
        except Exception:
            return False
        yaw = self.tf.transformations.euler_from_quaternion(rot)[2]
        self.pose = (trans[0], trans[1], yaw)
        self.pose_time = self.rospy.Time.now()
        return True

    def plan_from_current_pose(self, reason, force=False):
        now = self.rospy.Time.now()
        if (not force and
                (now - self.last_plan_time).to_sec() < self.replan_min_interval_s):
            return bool(self.path_world)
        if self.pose is None:
            return False

        self.last_plan_time = now
        start = (self.pose[0], self.pose[1])
        goal = (self.goal_x, self.goal_y)
        self.rospy.logwarn(
            "GlobalFirstGraphNav: planning %s from (%.2f, %.2f) to (%.2f, %.2f)",
            reason, start[0], start[1], goal[0], goal[1])
        result = self.planner.plan(start, goal)
        if not result["ok"]:
            self.replan_fail_count += 1
            self.path_world = []
            self.publish_zero("PLAN_FAIL")
            self.log_status("plan failed: %s" % result["error"])
            return False

        self.replan_fail_count = 0
        self.path_world = result["path_world"]
        self.path_index = 0
        self.active_goal = result["active_goal_world"]
        self.last_progress_time = now
        self.last_progress_pose = start
        self.last_progress_yaw = self.pose[2] if self.pose is not None else None
        self.last_goal_dist = self.distance_to_active_goal()
        self.publish_path()
        self.log_status(
            "plan ok: method=%s roadmap_nodes=%d points=%d length=%.2fm min_clear=%.2fm clearance_used=%.2fm" %
            (result["planner_method"], result["roadmap_nodes"], len(self.path_world),
             result["path_length_m"], result["min_clearance_m"],
             result["clearance_used_m"]))
        return True

    def publish_initial_pose(self, force=False):
        if not self.publish_initial_pose_on_start:
            return
        now = self.rospy.Time.now()
        if self.initial_pose_start_time is None:
            self.initial_pose_start_time = now
        if not force:
            if (now - self.initial_pose_start_time).to_sec() > self.initial_pose_repeat_s:
                return
            if (now - self.initial_pose_last_pub).to_sec() < self.initial_pose_period_s:
                return

        from geometry_msgs.msg import PoseWithCovarianceStamped
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = now
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = self.initial_pose_x
        msg.pose.pose.position.y = self.initial_pose_y
        qx, qy, qz, qw = quat_from_yaw(self.initial_pose_yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0] = 0.08 * 0.08
        msg.pose.covariance[7] = 0.08 * 0.08
        msg.pose.covariance[35] = math.radians(12.0) ** 2
        self.initial_pose_pub.publish(msg)
        self.initial_pose_last_pub = now

    def start_pose_ok(self):
        if self.start_pose_accepted:
            return True
        self.publish_initial_pose()
        if self.pose is None:
            return False
        dx = self.pose[0] - self.initial_pose_x
        dy = self.pose[1] - self.initial_pose_y
        dist = math.hypot(dx, dy)
        yaw_err = abs(norm_angle(self.pose[2] - self.initial_pose_yaw))
        yaw_limit = math.radians(self.start_yaw_tolerance_deg)
        if dist <= self.start_pose_tolerance_m and yaw_err <= yaw_limit:
            self.start_pose_accepted = True
            self.log_status(
                "initial pose accepted: pose=(%.2f, %.2f, %.1fdeg) dist=%.2fm" %
                (self.pose[0], self.pose[1], math.degrees(self.pose[2]), dist))
            return True
        self.rospy.logwarn_throttle(
            1.5,
            "GlobalFirstGraphNav: waiting for initial pose near (%.2f, %.2f, %.1fdeg); current=(%.2f, %.2f, %.1fdeg), dist=%.2fm",
            self.initial_pose_x, self.initial_pose_y, math.degrees(self.initial_pose_yaw),
            self.pose[0], self.pose[1], math.degrees(self.pose[2]), dist)
        return False

    def control_loop(self, _event):
        if self.finished:
            self.publish_zero("FINISHED")
            return
        if self.grid is None or self.planner is None:
            self.publish_zero("NO_MAP")
            return
        if not self.pose_fresh():
            self.publish_zero("NO_POSE")
            return
        if not self.start_pose_ok():
            self.publish_zero("WAIT_INITIAL_POSE")
            return
        if not self.scan_fresh():
            self.publish_zero("NO_SCAN")
            return
        if not self.path_world:
            if not self.plan_from_current_pose("initial", force=True):
                return

        if self.check_goal():
            return
        self.update_progress()
        if (self.rospy.Time.now() - self.last_progress_time).to_sec() > self.stuck_timeout_s:
            self.log_status("stuck watchdog: replanning instead of ending task")
            self.plan_from_current_pose("stuck", force=True)
            return

        target = self.select_target()
        if target is None:
            self.plan_from_current_pose("path exhausted", force=True)
            return
        cmd = self.compute_cmd(target)
        cmd = self.apply_scan_guard(cmd)
        cmd = self.smooth_cmd(cmd)
        self.publish_cmd(cmd[0], cmd[1])

    def check_goal(self):
        dist = self.distance_to_active_goal()
        if dist > self.goal_tolerance:
            self.goal_enter_time = None
            return False

        yaw_ok = True
        if self.require_goal_yaw:
            yaw_err = norm_angle(self.goal_yaw - self.pose[2])
            yaw_ok = abs(yaw_err) <= self.goal_yaw_tolerance
            if not yaw_ok:
                if min(self.left, self.right) < self.side_stop_m:
                    self.publish_zero("GOAL_YAW_SIDE_CLOSE")
                else:
                    wz = clamp(1.0 * yaw_err, -0.30, 0.30)
                    self.publish_cmd(0.0, wz)
                return True

        now = self.rospy.Time.now()
        if self.goal_enter_time is None:
            self.goal_enter_time = now
            self.publish_zero("GOAL_HOLD")
            return True
        if yaw_ok and (now - self.goal_enter_time).to_sec() >= self.goal_hold_s:
            self.finished = True
            self.publish_zero("GOAL_REACHED")
            self.log_status("goal reached and held; node stays alive with zero cmd")
            return True
        self.publish_zero("GOAL_HOLD")
        return True

    def distance_to_active_goal(self):
        if self.pose is None:
            return INF
        return math.hypot(self.pose[0] - self.active_goal[0],
                          self.pose[1] - self.active_goal[1])

    def update_progress(self):
        now = self.rospy.Time.now()
        current = (self.pose[0], self.pose[1])
        goal_dist = self.distance_to_active_goal()
        moved = 0.0
        if self.last_progress_pose is not None:
            moved = math.hypot(current[0] - self.last_progress_pose[0],
                               current[1] - self.last_progress_pose[1])
        improved = self.last_goal_dist - goal_dist
        yaw_changed = 0.0
        if self.last_progress_yaw is not None:
            yaw_changed = abs(norm_angle(self.pose[2] - self.last_progress_yaw))
        if (moved >= self.progress_dist_m or
                improved >= self.progress_dist_m or
                yaw_changed >= 0.18):
            self.last_progress_pose = current
            self.last_progress_yaw = self.pose[2]
            self.last_goal_dist = goal_dist
            self.last_progress_time = now

    def select_target(self):
        if not self.path_world:
            return None
        x, y, _ = self.pose
        lo = max(0, self.path_index - 8)
        hi = min(len(self.path_world), self.path_index + 120)
        best = self.path_index
        best_d2 = INF
        for i in range(lo, hi):
            px, py = self.path_world[i]
            d2 = (px - x) * (px - x) + (py - y) * (py - y)
            if d2 < best_d2:
                best_d2 = d2
                best = i
        self.path_index = max(self.path_index, best)

        accum = 0.0
        prev = (x, y)
        for i in range(self.path_index, len(self.path_world)):
            pt = self.path_world[i]
            accum += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            prev = pt
            if accum >= self.lookahead_dist:
                return pt
        return self.path_world[-1]

    def compute_cmd(self, target):
        x, y, yaw = self.pose
        dx = target[0] - x
        dy = target[1] - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading = math.atan2(local_y, max(local_x, 1.0e-4))
        if local_x < -0.04:
            heading = math.atan2(local_y, local_x)
        abs_h = abs(heading)

        if abs_h > self.rotate_in_place_angle or local_x < -0.04:
            can_creep = (
                local_x > -0.12 and
                self.front > self.turn_creep_front_min and
                min(self.left, self.right) > self.turn_creep_side_min)
            linear = self.turn_creep_speed if can_creep else 0.0
        else:
            scale = max(0.0, 1.0 - (abs_h / self.rotate_in_place_angle) ** 1.4)
            linear = self.max_linear * scale
            if linear > 0.0:
                linear = max(self.min_tracking_speed, linear)
        angular = self.k_heading * heading + self.k_lateral * local_y
        angular = clamp(angular, -self.max_angular, self.max_angular)
        return linear, angular

    def apply_scan_guard(self, cmd):
        linear, angular = cmd
        linear = max(0.0, linear)
        now = self.rospy.Time.now()

        if self.front < self.front_stop_m:
            if self.front_block_start is None:
                self.front_block_start = now
            blocked_for = (now - self.front_block_start).to_sec()
            if (blocked_for >= self.front_replan_after_s and
                    (now - self.last_blocked_replan).to_sec() >= self.blocked_replan_s):
                self.last_blocked_replan = now
                self.log_status(
                    "front blocked %.2fm for %.1fs: global graph replan requested" %
                    (self.front, blocked_for))
                self.plan_from_current_pose("front blocked", force=True)

            if self.left < self.side_stop_m and self.right < self.side_stop_m:
                # Deadlocked: front blocked + both sides tight.
                # Turn slowly toward the side with more room to wiggle out.
                wiggle_wz = 0.08
                if self.left > self.right:
                    return 0.0, wiggle_wz
                else:
                    return 0.0, -wiggle_wz
            turn_sign = 1.0 if self.left >= self.right else -1.0
            if turn_sign > 0.0 and self.left < self.side_stop_m:
                turn_sign = -1.0
            if turn_sign < 0.0 and self.right < self.side_stop_m:
                turn_sign = 1.0
            desired_turn = turn_sign * self.front_turn_speed
            if angular * desired_turn <= 0.0 or abs(angular) < abs(desired_turn):
                angular = desired_turn
            return 0.0, clamp(angular, -self.front_turn_speed, self.front_turn_speed)

        self.front_block_start = None

        if self.front < self.front_slow_m:
            t = (self.front - self.front_stop_m) / max(
                self.front_slow_m - self.front_stop_m, 1.0e-3)
            linear *= clamp(t, 0.18, 1.0)
            angular *= clamp(t, 0.40, 1.0)

        side_min = min(self.left, self.right)
        if side_min < self.side_stop_m:
            linear = min(linear, 0.025)
            if angular > 0.0 and self.left < self.side_stop_m:
                angular = 0.0
            if angular < 0.0 and self.right < self.side_stop_m:
                angular = 0.0
        elif side_min < self.side_slow_m:
            t = (side_min - self.side_stop_m) / max(
                self.side_slow_m - self.side_stop_m, 1.0e-3)
            ratio = clamp(t, 0.25, 1.0)
            linear *= ratio
            angular *= max(0.45, ratio)

        return linear, clamp(angular, -self.max_angular, self.max_angular)

    def smooth_cmd(self, cmd):
        now = self.rospy.Time.now()
        dt = max((now - self.last_control_time).to_sec(), 1.0 / self.control_rate_hz)
        self.last_control_time = now
        vx, wz = cmd
        last_vx, last_wz = self.last_cmd
        max_dv = self.linear_accel * dt
        max_dw = self.angular_accel * dt
        vx = clamp(vx, last_vx - max_dv, last_vx + max_dv)
        wz = clamp(wz, last_wz - max_dw, last_wz + max_dw)
        vx = clamp(vx, 0.0, self.max_linear)
        wz = clamp(wz, -self.max_angular, self.max_angular)
        self.last_cmd = (vx, wz)
        return vx, wz

    def publish_cmd(self, vx, wz):
        msg = self.Twist()
        msg.linear.x = max(0.0, vx)
        msg.angular.z = wz
        self.cmd_pub.publish(msg)

    def publish_zero(self, reason):
        self.last_cmd = (0.0, 0.0)
        self.publish_cmd(0.0, 0.0)
        self.rospy.logwarn_throttle(2.0, "GlobalFirstGraphNav zero: %s", reason)

    def publish_path(self):
        from geometry_msgs.msg import PoseStamped
        msg = self.Path()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = self.map_frame
        for i, (x, y) in enumerate(self.path_world):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            if i + 1 < len(self.path_world):
                nx, ny = self.path_world[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = self.goal_yaw
            qx, qy, qz, qw = quat_from_yaw(yaw)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def log_status(self, text):
        self.rospy.logwarn("GlobalFirstGraphNav: %s", text)
        msg = self.String()
        msg.data = text
        self.status_pub.publish(msg)

    def shutdown(self):
        """Graceful shutdown: stop motors and clean up."""
        self.rospy.logwarn("GlobalFirstGraphNav: shutting down...")
        self.finished = True
        # Publish zero velocity to stop the robot
        self.publish_zero("SHUTDOWN")
        self.rospy.logwarn("GlobalFirstGraphNav: shutdown complete")

    def spin(self):
        self.rospy.spin()


def ros_main():
    node = RosGlobalFirstGraphNavigator()
    try:
        node.spin()
    except KeyboardInterrupt:
        node.rospy.logwarn("GlobalFirstGraphNav: Ctrl+C received, exiting gracefully")
        node.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--offline-check":
        sys.exit(offline_main(sys.argv[2:]))
    ros_main()
