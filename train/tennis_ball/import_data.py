from roboflow import Roboflow

rf = Roboflow(api_key="oDXoSu50E8lBlOmEtp2y")
project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolov5")
                