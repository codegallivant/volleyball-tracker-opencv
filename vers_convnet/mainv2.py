import cv2 as cv
from inference_sdk import InferenceHTTPClient
import numpy as np
from inference import get_roboflow_model
import supervision as sv

# define the image url to use for inference

# load a pre-trained model
model = get_roboflow_model(model_id="volleyball-tracking/18")

vid = cv.VideoCapture("volleyball_match.mp4")
length = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
fps = vid.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'mp4v') 
video = cv.VideoWriter('volleyball_detected.mp4', fourcc, fps, (1280, 720))

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="dGlsQHSH4oeasH91KcDf"
# )
#  {'time': 0.0697369349999235, 'image': {'width': 1280, 'height': 720}, 'predictions': [{'x': 954.375, 'y': 181.875, 'width': 18.75, 'height': 18.75, 'confidence': 0.786199688911438, 'class': 'volleyball', 'class_id': 0, 'detection_id': 'bc26f528-c0b2-4a0e-a96c-1a498c924963'}, {'x': 495.0, 'y': 127.5, 'width': 17.5, 'height': 15.0, 'confidence': 0.7268819808959961, 'class': 'volleyball', 'class_id': 0, 'detection_id': 'b8797aa2-265b-460a-a4db-fbf6464fbcab'}]}
# export ROBOFLOW_API_KEY= <redacted>

i = 0
for fn in range(0, length):
    # if i > 500:
    #     break
    ret, frame = vid.read()
    if ret:
        print(f"{'%2.f'%(((i+1)/length)*100)}% complete ({i+1}/{length} frames)")
        # result = CLIENT.infer(frame, model_id="volleyball-tracking/18")
        result = model.infer(frame)[0].dict(by_alias=True, exclude_none = True)
        # results[0].dict(by_alias=True, exclude_none=True)
        print(result)
        first = True
        for pred in result["predictions"]:
            conf = pred['confidence']
            if conf >= 0.5:
                if first == True:
                    colour = (255,255,255)
                    first = False
                else:
                    colour = (128,128,128)
                width = pred['width']
                height = pred['height']
                x = int(pred['x'])
                y = int(pred['y'])
                frame = cv.rectangle(frame, (x-int(width/2),y-int(height/2)), (x+int(width/2),y+int(height/2)), colour, 2)
                frame = cv.putText(frame, '%.2f'%(pred['confidence']*100), (x+int(width/2),y-int(height/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA) 
        cv.imshow("Tracking", frame)
        video.write(frame)
        i = i + 1
        key = cv.waitKey(1)
        if key == 27:
            break

video.release()
cv.destroyAllWindows()


