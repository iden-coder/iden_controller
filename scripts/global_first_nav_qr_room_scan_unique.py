#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Original QR-room scanner with stable physical-code deduplication."""

import os
import sys
from urllib.parse import urlsplit

import rospy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_first_nav_qr_room_scan import GlobalFirstNavThenQR  # noqa: E402


class UniqueQRRoomScanner(GlobalFirstNavThenQR):
    def __init__(self):
        super(UniqueQRRoomScanner, self).__init__()
        self.qr_identity_to_wall = {}

    @staticmethod
    def stable_qr_identity(raw):
        text = (raw or "").strip()
        if not text:
            return ""
        if text.startswith(("http://", "https://")):
            parsed = urlsplit(text)
            path = parsed.path.rstrip("/") or "/"
            # The endpoint URL identifies the printed QR.  Its HTTP response
            # may intentionally return a different random item on every scan.
            return "url:{}://{}{}".format(
                parsed.scheme.lower(), parsed.netloc.lower(), path)
        return "raw:" + "".join(text.split())

    def result_has_item_payload(self, result):
        if isinstance(result, dict) and result.get("duplicate_qr"):
            return False
        return super(UniqueQRRoomScanner, self).result_has_item_payload(result)

    def store_wall_result(self, result):
        identity = self.stable_qr_identity(result.get("raw", ""))
        if identity and identity in self.qr_identity_to_wall:
            first_wall = self.qr_identity_to_wall[identity]
            result["duplicate_qr"] = True
            result["duplicate_of_wall"] = first_wall
            rospy.logwarn(
                "QR_DUPLICATE_IGNORED wall=%s first_wall=%s identity=%s; continuing scan",
                result.get("wall_index"), first_wall, identity)
            return False

        accepted = super(UniqueQRRoomScanner, self).store_wall_result(result)
        if accepted and identity:
            self.qr_identity_to_wall[identity] = result.get("wall_index")
            rospy.loginfo(
                "QR_UNIQUE_ACCEPTED wall=%s unique=%d/%d identity=%s",
                result.get("wall_index"), len(self.qr_identity_to_wall),
                self.target_qr_count, identity)
        return accepted


def main():
    node = UniqueQRRoomScanner()
    node.run()
    rospy.loginfo("unique QR room scan finished; node stays alive")
    rospy.spin()


if __name__ == "__main__":
    main()

