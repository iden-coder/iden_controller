#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure formatter for the only voice sentence allowed in the large room."""


def delivery_voice(item, warehouse):
    item = str(item or "货品").strip() or "货品"
    warehouse = str(warehouse or "目标车间").strip() or "目标车间"
    return "已将{}放入{}".format(item, warehouse)

