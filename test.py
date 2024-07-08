import ultralyticsplus
from ultralyticsplus import YOLO, render_result


model = YOLO('keremberke/yolov8m-table-extraction')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
image = 'C:/Works/ISD/test/pdf/table-word.jpg'

# perform inference
results = model.predict(image)

# observe results
print(results)
render = render_result(model=model, image=image, result=results[0])
render.show()