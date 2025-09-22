import cv2
import os
import numpy as np

video_path = "Minecraft_stitch_test.mp4"
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

KEEP_EVERY = 10
USE_SCANS = True

def frame_reader(video_path, keep_every=10, save_dir=None):
    cap = cv2.VideoCapture(video_path)

    images = []
    i = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % keep_every == 0:
            images.append(frame)
            if save_dir is not None:
                out_path = os.path.join(save_dir, f"frame_{i:05d}.jpg")
                cv2.imwrite(out_path, frame)
            ok_write = cv2.imwrite(out_path, frame)
            if ok_write:
                saved += 1
            else:
                print(f"[WARN] Failed to write: {out_path}")


        i += 1
    cap.release()
    return images


def stitch(images):
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("Not enough images for stitching")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Homography estimation failed")
    else:
        print("Image stitching failed")

frame_reader(video_path, KEEP_EVERY, output_folder)
