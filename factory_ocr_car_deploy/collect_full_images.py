import cv2
import time
import os
import argparse


def get_next_idx(save_dir, label):
    ids = []
    for name in os.listdir(save_dir):
        if not name.lower().endswith(('.jpg', '.png')):
            continue
        stem = os.path.splitext(name)[0]
        parts = stem.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            ids.append(int(parts[-1]))
    return max(ids) + 1 if ids else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, choices=['food', 'daily', 'electric', 'unknown'])
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--interval', type=float, default=0.3)
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args()

    save_dir = f'./full_cls_collect/{args.label}'
    os.makedirs(save_dir, exist_ok=True)

    idx = get_next_idx(save_dir, args.label)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print('video0 打不开，尝试 video1')
        cap.release()

        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print('摄像头打开失败')
            return

    print('开始采集整张图')
    print('类别:', args.label)
    print('保存目录:', save_dir)
    print('按 Ctrl+C 停止')
    print('当前起始编号:', idx)

    last_save = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print('读取摄像头失败')
                time.sleep(0.1)
                continue

            if args.flip:
                frame = cv2.flip(frame, 1)

            now = time.time()
            if now - last_save < args.interval:
                continue

            filename = f'{args.label}_{idx:06d}.jpg'
            path = os.path.join(save_dir, filename)
            cv2.imwrite(path, frame)

            print('saved:', path)

            idx += 1
            last_save = now

    except KeyboardInterrupt:
        print('停止采集')

    cap.release()


if __name__ == '__main__':
    main()
