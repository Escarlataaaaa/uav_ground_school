import cv2
import os
import numpy as np

video_path = "Minecraft_stitch_test.mp4"
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

KEEP_EVERY = 10
USE_SCANS = True
OUT_PATH   = "map.jpg"

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
                if cv2.imwrite(out_path, frame):
                    saved += 1
                else:
                    print(f"[WARN] failed to write: {out_path}")
        i += 1
    cap.release()
    return images

def sift_extractor():
    return cv2.SIFT_create(nfeatures=5000)
    
def detect_and_describe(image, sift):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_descriptors(descA, descB, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(descA, descB, k=2)
    return [m for m,n in raw if m.distance < ratio * n.distance]

def stitch(images, scans=False):
    mode = cv2.Stitcher_SCANS if scans else cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher_create(mode)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return pano
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("Not enough images for stitching")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Homography estimation failed")
    else:
        print("Image stitching failed")


if __name__ == "__main__":
    frames = frame_reader(video_path, keep_every=KEEP_EVERY, save_dir=output_folder)
    pano = stitch(frames, scans=USE_SCANS)
    cv2.imwrite(OUT_PATH, pano)
    print(f"[DONE] saved mosaic: {os.path.abspath(OUT_PATH)}")