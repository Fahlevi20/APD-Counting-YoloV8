from flask import Flask, request, render_template
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os