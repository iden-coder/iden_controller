#!/usr/bin/env python3

"""Pure decisions for the room-level voice fail-safe watchdog."""


def should_fire_watchdog(active, aborted, deadline, now):
    return bool(active and not aborted and deadline is not None and now >= deadline)


def error_voice_text(reason=None):
    del reason
    return "任务执行异常，小车已安全停止"
