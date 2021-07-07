from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

#img -> image_data
#im0s -> normal
def detect(img,im0s):
    img_size = (128,128)
    weights, half= 'weights/frozenBest.pt', True

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else '')

    # Initialize model
    model = Darknet('cfg/yolov3-spp-r.cfg', img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()



    # Get detections
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]
    

    # Apply NMS
    pred = non_max_suppression(pred, 0.01, 0.01)


    validDetections = 0
    # Apply
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        bboxes= []
        scores= []
        classes= []
        num_objects = 0
        #pred
        #is a list of det
        #det
        #Can change to list by .toList
        # Xmin ymin xmax ymax confidence blank class
        #[tensor([[182.05904, 107.74407, 247.58714, 355.46249,   0.44077,   0.99997,   0.00000]])]
        # det
        # [[178.6617431640625, 105.31470489501953, 245.80511474609375, 360.6492919921875, 0.8271918892860413, 0.9999979734420776, 0.0]
        #,[0.3538074493408203, 249.970703125, 41.941795349121094, 413.47216796875, 0.7629737854003906, 0.9999510049819946, 0.0]]
        if det != None:
            buffer = det.tolist()
            num_objects = len(det.tolist())
            bboxes= []
            scores= []
            classes= []
            for i in range(num_objects):
                if (buffer[i][2]-buffer[i][0])*(buffer[i][3]-buffer[i][1]) > 0 and buffer[i][6] == 0.0:
                    if buffer[i][4] > 0.5:
                        validDetections+=1
                        #insert minimum bounding box size here 
                        bboxes.append(buffer[i][:4])
                        scores.append(buffer[i][4])
                        classes.append(buffer[i][6])

            print("len")
            print(validDetections)
            print('bbox')
            print(bboxes)
            print('scores')
            print(scores)
            print('classes')
            print(classes)

    return (validDetections,bboxes,scores,classes)


