!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="3Irr39DHWqAkEhIqAHU9")
project = rf.workspace("road-damage-detection-n2xkq").project("crack-and-pothole-bftyl")
version = project.version(1)
dataset = version.download("yolov10")
