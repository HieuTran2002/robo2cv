import cv2
import inference
import supervision as sv


annotator = sv.BoxAnnotator()



def on_prediction(predictions, image):
    # labels = [p["class"] for p in predictions["predictions"]]

    print(len(predictions['predictions']))

    for bounding_box in predictions['predictions']:
        x0 = (int)(bounding_box['x'] - bounding_box['width'] / 2)
        x1 = (int)(bounding_box['x'] + bounding_box['width'] / 2)
        y0 = (int)(bounding_box['y'] - bounding_box['height'] / 2)
        y1 = (int)(bounding_box['y'] + bounding_box['height'] / 2)

        start_point = (x0, y0)
        end_point = (x1, y1)
        cv2.rectangle(image,
                      start_point,
                      end_point,
                      color=(0, 250, 0),
                      thickness=5)

        # print(f"point 1: {x0}:{y0} | point 2: {x1}:{y1}")
        cropped = image.copy()[y0:y1, x0:x1]
        # cv2.imshow("crop", cropped)

        cv2.putText(
            image,
            bounding_box["class"],
            (int(x0), int(y0) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(255, 255, 255),
            thickness=2
        )

    cv2.imshow("result", image)
    cv2.waitKey(1)


inference.Stream(
    # replace with your camera IP
    source="2.mp4",
    model="vflame_robot2/2",
    output_channel_order="BGR",
    use_main_thread=True,  # for opencv display
    on_prediction=on_prediction,
)
