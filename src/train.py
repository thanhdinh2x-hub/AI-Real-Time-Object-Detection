import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights,FastRCNNPredictor,FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from datasetVOC2007 import VOCdataset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm  #autonotebook thì chạy đc cả trong notebook cả trong bình thường
from torch.utils.tensorboard import SummaryWriter
import shutil # TODO : Nếu muốn xóa cả folder log cũ thay os.rmdir: cái này chỉ xóa file rỗng đc thôi
import os
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint


def collate_fn(batch):  # defined cách trả ra list của dataloader
    # print(batch)
    all_images = []
    all_labels = []
    for image, labels in batch:
        # print(image.shape)
        # print(labels)
        all_images.append(image)
        all_labels.append(labels)
        # print(len(all_images))
        # print(all_labels)
    return all_images, all_labels


def get_args():
    parser = ArgumentParser(description='CNN Argument Parser')
    parser.add_argument('--root','-r', help='root path', default="../datasets", type=str)

    parser.add_argument('--num_epochs','-ne', help='number of epochs', default=100, type=int)
    parser.add_argument('--batch_size','-b', help='batch size', default=3, type=int)
    parser.add_argument('--image_size','-ims', help='image size', default=416, type=int)

    parser.add_argument('--logging','-l', help='logging tesorboard', default="tensorboard", type=str)
    parser.add_argument('--checkpoint','-c', help='checkpoint is models saved params', default="../trained_models/last_model.pt", type=str)
    parser.add_argument('--save_path','-sp', help='save_path is path where models saved params', default="../trained_models", type=str)

    argus = parser.parse_args()
    return argus

def train(args):
    #TODO-Step 1: Setup a dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            # đây là chuẩn hóa trên từng kênh màu của toàn bộ bức ảnh trong epoch các mô hình transfer thường theo tiêu chuẩn này
            std=[0.229, 0.224, 0.225]),
    ])
    val_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            # đây là chuẩn hóa trên từng kênh màu của toàn bộ bức ảnh trong epoch các mô hình transfer thường theo tiêu chuẩn này
            std=[0.229, 0.224, 0.225]),
    ])

    train_set = VOCdataset(root=args.root, year="2007", image_set="trainval", download=False,
                           transform=train_transform)
    val_set = VOCdataset(root=args.root, year="2007", image_set="val", download=False, transform=val_transform)

    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 4,  # càng nhiều thì càng có worker nhảy vào datasets để cb sẵn data
        "drop_last": True,
        "collate_fn": collate_fn,
    }
    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 4,
        "drop_last": False,
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # TODO-Step 2: đefined model dùng để  transfer learning
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,trainable_backbone_layers=6 ) # trainable_backbone_layers nghia là ko đóng băng layer nào
    # # region print roi_head details
    # print(model)
    # print("---------")
    # print(model.backbone)
    # print("---------")
    # print(model.roi_heads)
    # print("---------")
    # print(model.roi_heads.box_predictor)
    # print("---------")
    # print(model.roi_heads.box_predictor.cls_score)
    # print("---------")
    # print(model.roi_heads.box_predictor.cls_score.in_features)
    # # endregion
    # FIXME transfer learning nên thay dầu ra của model cũ
    print("----Thay box_predictor -----")
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes=len(train_set.categories))

    # print(model.roi_heads.box_predictor)
    model.to(device)
    # TODO-Step 3: Define a tensorboard and an optimizer,...
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,
                                momentum=0.9)  # define optimizer . models.parameters() là muốn update toàn bộ parmeter
    # num_interation=len(train_loader) # training_loader chỉ là 1 list chứa các batch data thôi
    # FIXME nếu tensorboard tồn tại thì xóa tensorboard cũ đi
    if os.path.exists(args.logging):
        shutil.rmtree(args.logging)
    os.makedirs(args.logging)
    # FIXME nếu trained_models ko tồn tại thì dic mới để lưu params của model về sau
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    # FIXME khởi tạo thi viện để vẽ tensorboard
    writer = SummaryWriter(log_dir=args.logging)

    best_map = 0 # define best_map
    # kiểm tra xem có lưu thông tin epoch trc ko
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint) # load đường dẫn
        start_epoch=checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state_dict"])
        best_map = checkpoint['best_map']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_epoch=0
    # TODO-Step 4: Train model từng epoch
    for epoch in range(start_epoch ,args.num_epochs):
        # FIXME-train step
        model.train()
        train_loss_list = []
        train_progress_bar = tqdm(train_loader,colour="cyan")
        for iter,(images, targets) in enumerate(train_progress_bar):

            # region print len(images)/targets details
            # print("len(images):",len(images))
            # print(targets)
            '''len(images): 3
            [{'boxes': tensor([[ 39., 118., 398., 289.]]), 'labels': tensor([1])}, {'boxes': tensor([[ 34.,  85., 216., 373.],
            [212.,  90., 416., 416.]]), 'labels': tensor([15, 15])}, {'boxes': tensor([[ 46.,  44., 307., 403.],
            [ 67.,  48., 203., 264.]]), 'labels': tensor([14, 15])}]
            '''
            # endregion

            # FIXME biến tất cả ảnh trong images thành to(device) để chạy trên gpu
            images = [image.to(device) for image in images]
            #FIXME đi vào từng phần tử dictionary -> biến từng value(là các list tensor ) của dic thành to(device) để chạy trên gpu
            targets = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in targets]
            loss_components = model(images, targets)
            # print(loss_components)
            # print(loss_components.values())
            # region print loss_components details
            '''
               {'loss_classifier': tensor(2.9806, device='cuda:0', grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.1390, device='cuda:0', grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.1944, device='cuda:0',
                      grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0179, device='cuda:0', grad_fn=<DivBackward0>)}
               dict_values([tensor(2.9806, device='cuda:0', grad_fn=<NllLossBackward0>), tensor(0.1390, device='cuda:0', grad_fn=<DivBackward0>), tensor(0.1944, device='cuda:0',
                      grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), tensor(0.0179, device='cuda:0', grad_fn=<DivBackward0>)])
               '''
            # endregion

            losses= sum([ loss for loss in loss_components.values()]) #loss sẽ trả ra từng giá trị và sum sẽ cộng từng giá trị đó

            # FIXME-Backward and Optimize
            optimizer.zero_grad()  # FIXME- 1. Xóa gradient cũ( ở đây có thề là của batch trước)- xóa hết bộ nhớ về gradient đi vì chưa cần lm vc với video
            losses.backward()  # 2. Tính gradient mới dựa vào loss
            optimizer.step()  # 3. Dùng gradient để sửa weight quay lại update lại pameters

            # FIXME- muốn show ra giá trị losses ổn định hơn
            # print(losses.item())
            train_loss_list.append(losses.item())
            avg_train_loss = np.mean(train_loss_list)

            # FIXME- dùng tqdm thì phải cho in từng iteration để có thanh bar chạy
            train_progress_bar.set_description("Epoch [{}/{}]. Loss {:0.4f} ".format(epoch + 1, args.num_epochs,avg_train_loss))
            # FIXME- dùng tensorboard để ghi lại đồ thị loss khi trainning
            writer.add_scalar("Loss/train", avg_train_loss, global_step=epoch*len(train_loader) +iter )
        # FIXME-test step
        model.eval()

        metric = MeanAveragePrecision() #FIXME define metric mAp
        test_progress_bar = tqdm(val_loader, colour="green")
        for iter, (images, targets) in enumerate(test_progress_bar):
            # FIXME biến tất cả ảnh trong images thành to(device) để chạy trên gpu
            images = [image.to(device) for image in images]
            # FIXME đi vào từng phần tử dictionary -> biến từng value(là các list tensor ) của dic thành to(device) để chạy trên gpu
            targets = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in
                       targets]

            # FIXME-bắt đầu forward nhưng ko lưu trữ thông tin gradient
            with torch.no_grad():  # FIXME -Ngăn không tạo: thông tin phục vụ backward ngay từ lúc forward.
                prediction = model(images)
                # region print prediction details
                # print(prediction)
                # print(type(prediction))
                # print(len(prediction))
                '''
                [{'boxes': tensor([[1.3591e+02, 2.9248e+02, 2.1849e+02, 4.0947e+02],
                [5.4175e+00, 2.5691e+02, 5.9607e+01, 4.1292e+02],
                [2.1926e+02, 2.4432e+02, 2.5830e+02, 3.5438e+02],
                [2.2380e+02, 2.2376e+02, 2.6701e+02, 3.2254e+02],
                [2.2574e+02, 2.1189e+02, 2.6842e+02, 2.7547e+02],
                [7.3866e+01, 2.9269e+02, 1.7084e+02, 4.1600e+02],
                [2.3078e+02, 2.0645e+02, 2.6458e+02, 2.5258e+02],
                [5.5168e-01, 2.2692e+02, 5.6869e+01, 3.4782e+02],
                [2.4523e+02, 2.4323e+02, 3.1928e+02, 4.1265e+02],
                [2.2336e+02, 2.7064e+02, 2.4894e+02, 3.5450e+02],
                [1.8377e+02, 2.1395e+02, 2.2717e+02, 3.1356e+02],
                [2.9720e+02, 2.4116e+02, 3.9874e+02, 4.1111e+02],
                [2.3330e+02, 2.6307e+02, 2.5432e+02, 3.5523e+02],
                [5.1133e+01, 2.8292e+02, 2.4827e+02, 4.1326e+02],
                [0.0000e+00, 2.2502e+02, 3.5810e+01, 2.7837e+02],
                [2.1472e+02, 2.3295e+02, 2.8479e+02, 3.5147e+02],
                [1.7047e+02, 2.3668e+02, 3.9451e+02, 4.1352e+02],
                [2.0896e+02, 2.3067e+02, 2.4625e+02, 3.4076e+02],
                [1.0625e+02, 3.0385e+02, 1.5979e+02, 4.1279e+02],
                [1.5840e+02, 2.8477e+02, 2.9848e+02, 4.1196e+02],
                [2.8657e+01, 2.6960e+02, 1.5404e+02, 4.0957e+02],
                [2.2935e+02, 2.4032e+02, 2.6702e+02, 2.8141e+02],
                [1.8636e+02, 3.8369e+02, 2.9613e+02, 4.1563e+02],
                [2.0122e+02, 2.2123e+02, 2.3382e+02, 3.3007e+02],
                [6.2201e+01, 2.9181e+02, 1.5806e+02, 4.1580e+02],
                [2.0311e+02, 2.1820e+02, 2.5343e+02, 2.9382e+02],
                [2.8147e-01, 2.2170e+02, 5.9607e+01, 3.9449e+02],
                [2.0501e+02, 2.1133e+02, 2.2712e+02, 2.7599e+02],
                [1.4039e+02, 2.3872e+02, 2.2470e+02, 3.8710e+02],
                [2.9868e+02, 2.3818e+02, 4.0540e+02, 3.9718e+02],
                [2.1025e+02, 2.5234e+02, 2.3378e+02, 3.2206e+02],
                [1.9681e+00, 2.6665e+02, 1.9172e+02, 4.1425e+02],
                [1.9720e+02, 2.4441e+02, 2.7834e+02, 3.9583e+02],
                [1.5369e+02, 2.8933e+02, 2.2662e+02, 3.6982e+02],
                [2.2057e+02, 2.5057e+02, 2.4686e+02, 3.2770e+02],
                [2.0429e+02, 2.7434e+02, 2.2582e+02, 3.2759e+02],
                [2.1395e+02, 2.7340e+02, 2.3476e+02, 3.4788e+02],
                [2.4409e+02, 2.2076e+02, 3.1673e+02, 4.0221e+02],
                [1.5103e+02, 2.1169e+02, 2.3812e+02, 3.1779e+02],
                [2.2064e+02, 2.3162e+02, 2.4978e+02, 2.9830e+02],
                [1.9243e+02, 3.9485e+02, 2.4011e+02, 4.1365e+02],
                [2.3139e+02, 2.0452e+02, 2.5127e+02, 2.3639e+02],
                [2.2700e+02, 2.9946e+02, 2.4546e+02, 3.6646e+02],
                [1.7759e+02, 2.2966e+02, 4.1595e+02, 4.1600e+02],
                [2.4669e+02, 2.0403e+02, 2.6164e+02, 2.3715e+02],
                [2.0793e-01, 2.7243e+02, 2.2963e+01, 4.0912e+02],
                [2.1085e+02, 2.1824e+02, 2.3465e+02, 2.9237e+02],
                [3.8776e+01, 2.7429e+02, 7.8508e+01, 4.0999e+02],
                [1.8655e+02, 2.3207e+02, 4.0904e+02, 4.0975e+02]], device='cuda:0'), 'labels': tensor([ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
                 9,  9,  9,  9,  9,  9, 11,  9, 18,  9,  9, 20,  9, 11,  9,  9,  9,  9,
                 9, 20,  9,  9,  9,  9,  9, 11,  9,  9,  9,  9, 18], device='cuda:0'), 'scores': tensor([0.9733, 0.8497, 0.6417, 0.5767, 0.5177, 0.3813, 0.2867, 0.2593, 0.2543,
                0.2515, 0.2286, 0.2198, 0.2098, 0.2037, 0.2022, 0.1974, 0.1908, 0.1672,
                0.1599, 0.1405, 0.1315, 0.1304, 0.1296, 0.1292, 0.1254, 0.1238, 0.1122,
                0.1089, 0.1061, 0.1006, 0.0992, 0.0990, 0.0983, 0.0900, 0.0884, 0.0852,
                0.0806, 0.0768, 0.0764, 0.0720, 0.0595, 0.0580, 0.0576, 0.0570, 0.0561,
                0.0557, 0.0540, 0.0532, 0.0508], device='cuda:0')}, {'boxes': tensor([[118.9446,  75.5317, 415.9631, 404.3110],
                [104.0449,  27.7532, 416.0000, 407.3336]], device='cuda:0'), 'labels': tensor([7, 6], device='cuda:0'), 'scores': tensor([0.9281, 0.1030], device='cuda:0')}, {'boxes': tensor([[231.2818, 227.9998, 265.7174, 366.0016],
                [129.5494, 152.5403, 184.1289, 323.9823],
                [ 90.7189, 214.4220, 237.1338, 350.0290],
                [224.4212, 230.1751, 248.9416, 361.3651],
                [132.6762, 165.8620, 222.2800, 361.0389],
                [219.3092, 227.6710, 239.7288, 358.2898],
                [158.8491, 196.1225, 212.2911, 358.2635],
                [216.5648, 226.8483, 230.3833, 351.9608],
                [123.3384, 219.3138, 177.6353, 360.4200],
                [135.2619, 192.3357, 158.8105, 301.6520],
                [143.3717, 157.3097, 187.6463, 245.2276],
                [221.0735, 270.6820, 266.8428, 359.7033],
                [ 92.2056, 230.4959, 165.0269, 354.7512]], device='cuda:0'), 'labels': tensor([15, 15, 13, 15, 15, 15, 15, 15, 13, 15, 15,  2, 13], device='cuda:0'), 'scores': tensor([0.9577, 0.9560, 0.9450, 0.8239, 0.8098, 0.4024, 0.2080, 0.1213, 0.0867,
                0.0850, 0.0615, 0.0583, 0.0502], device='cuda:0')}]
                <class 'list'>
                3'''
                # endregion
                metric.update(prediction, targets)
        map = metric.compute()
        pprint(metric.compute())
        # region print metric.compute() details

        '''{'classes': tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20], dtype=torch.int32),
         'map': tensor(0.3826),
         'map_50': tensor(0.6794),
         'map_75': tensor(0.3983),
         'map_large': tensor(0.4834),
         'map_medium': tensor(0.2499),
         'map_per_class': tensor(-1.),
         'map_small': tensor(0.0616),
         'mar_1': tensor(0.3542),
         'mar_10': tensor(0.4977),
         'mar_100': tensor(0.5119),
         'mar_100_per_class': tensor(-1.),
         'mar_large': tensor(0.6154),
         'mar_medium': tensor(0.3767),
         'mar_small': tensor(0.1294)}'''
        # endregion
        # FIXME- dùng tensorboard để ghi lại đồ thị mAP
        writer.add_scalar("mAP/val", map["map"], global_step=epoch * len(train_loader) + iter)

        #FIXME lưu lại checkpoint sau khi chạy train và val qua 1 epoch
        checkpoint = {
            'epoch': epoch+1,
            "best_map": best_map,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        #FIXME nếu epoch  mà có map tốt nhất thì ghi đè luôn best_map ở checkpoint và lưu luôn vào best_model.pt
        if map["map"] > best_map:
            best_map = map["map"]
            checkpoint = {
                'epoch': epoch + 1,
                "best_map": best_map,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.save_path, "best_model.pt"))

        torch.save(checkpoint, os.path.join(args.save_path, "last_model.pt"))


if __name__ == '__main__':
    args = get_args()
    train(args)
