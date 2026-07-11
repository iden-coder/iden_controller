# Factory Room Delivery (new-file integration)

This integration leaves every previous navigation, QR, speech and OCR file
unchanged.  It adds a second-stage room mission after the existing Spark result.

## Runtime flow

1. Wake word and category speech are handled by the existing subtask-1 node.
2. Existing sixth-version navigation reaches the QR room and scans all 3 QR codes.
3. Spark selects the item and the existing TTS announces the selection.
   QR navigation/rotation parameters remain identical to the established
   real-factory route; repeated scans of the same stable QR URL are ignored.
4. The new room manager waits 7 seconds, then navigates to `start`.
5. A circular dynamic costmap avoids cones while visiting d1-d3 first, then
   automatically generated center/midpoint/ring viewpoints when needed.
6. At each reachable observation point, RKNN OCR scans the walls in 45-degree steps.
7. The target workshop sign is centred, then the planner approaches its wall.
8. The camera detects the 50 cm white floor frame and performs a low-speed entry.
9. Laser distance, visual centring and odometry jointly verify parking before TTS says:
   `已将[货品名称]放入[仓库类别]`.

## Start

```bash
~/instant_ws/src/iden_controller/scripts/start_subtask1_factory_delivery_complete.sh
```

Emergency stop remains `Ctrl-C`.  The manager publishes zero velocity repeatedly
on shutdown, timeout or any failed safety check.

## Useful diagnostics

```bash
rostopic echo /factory/room_task_state
rostopic echo /factory_room/ocr_result
rostopic echo /factory/delivery_result
rqt_image_view /factory_room/ocr_debug
```

The first real trial should be run with the robot lifted or with a hand on the
physical emergency stop, then with no cones, and only then with cones added.
