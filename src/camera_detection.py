import numpy as np
import cv2
import torch

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FastRCNNPredictor
)


def preprocess(image, image_size, device):
    """
    OpenCV BGR image
    ->
    Tensor [3,H,W]
    """

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # lưu kích thước gốc để scale bbox lại
    h, w, _ = image.shape

    # resize về input model
    image = cv2.resize(
        image,
        (image_size, image_size)
    )

    # normalize giống lúc train
    image = image / 255.0

    image -= np.array(
        [0.485, 0.456, 0.406]
    )

    image /= np.array(
        [0.229, 0.224, 0.225]
    )


    # HWC -> CHW
    image = np.transpose(
        image,
        (2,0,1)
    )


    image = torch.from_numpy(image)
    image = image.float()
    image = image.to(device)


    # Faster R-CNN cần list Tensor
    return [image], w, h



def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    categories = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]


    image_size = 416
    confidence_threshold = 0.77


    # =========================
    # Load model
    # =========================

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )


    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=
        model.roi_heads.box_predictor.cls_score.in_features,

        num_classes=len(categories)
    )


    checkpoint = torch.load(
        "../trained_models/best_model.pt"
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.to(device)
    model.eval()


    # =========================
    # Open camera
    # =========================

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        print("Không mở được camera")
        return



    while True:

        ret, frame = cap.read()

        if not ret:
            break



        # preprocess
        images, width, height = preprocess(
            frame,
            image_size,
            device
        )


        # inference
        with torch.no_grad():

            prediction = model(images)



        # lấy output
        boxes = prediction[0]["boxes"]
        scores = prediction[0]["scores"]
        labels = prediction[0]["labels"]



        for box, score, label in zip(
            boxes,
            scores,
            labels
        ):


            if score > confidence_threshold:


                xmin, ymin, xmax, ymax = box


                # model dự đoán trên ảnh 416x416
                # scale về ảnh camera thật

                xmin = int(
                    xmin / image_size * width
                )

                ymin = int(
                    ymin / image_size * height
                )

                xmax = int(
                    xmax / image_size * width
                )

                ymax = int(
                    ymax / image_size * height
                )


                label_name = categories[label]


                cv2.rectangle(
                    frame,
                    (xmin,ymin),
                    (xmax,ymax),
                    (0,255,0),
                    2
                )


                cv2.putText(
                    frame,
                    f"{label_name}:{score:.2f}",

                    (xmin,ymin-10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (255,0,255),

                    2
                )



        cv2.imshow(
            "Faster R-CNN Camera",
            frame
        )


        # ESC để thoát
        if cv2.waitKey(1) == 27:
            break



    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()