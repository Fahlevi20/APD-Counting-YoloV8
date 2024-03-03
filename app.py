from flask import Flask, request, render_template
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os

#deploy model
model = YOLO("yolov8n_v1_train2\weights\best.pt")

#load input video
cap = cv2.VideoCapture("input\video contoh trim.mp4")
