from roboflow import Roboflow
rf = Roboflow(api_key="HmGsTmcZ2F9sxcskLknp")
project = rf.workspace().project("vflame_robot2")
model = project.version(2).model

# infer on a local image
print(model.predict("photo.jpg", confidence=40, overlap=30).json())
