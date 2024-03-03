from flask import Flask, request, render_template
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os

#deploy model
model = YOLO("yolov8n_v1_train2/weights/best.pt")

#load input video
cap = cv2.VideoCapture("input/video contoh trim.mp4")

assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)) #for set the size and fps

region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)] #for the range when we can counting the object

classes_to_count = [0, 1,2,3,4,5,6,7,8,9]  # person and car classes for count

video_writer = cv2.VideoWriter("output/object_counting_output_test_dengan.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# for counting the object
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False,
                         classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()