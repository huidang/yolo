# yolo
pytorch and python based yolov3
you can access the paper in https://pjreddie.com/media/files/papers/YOLOv3.pdf
This is pytorch based code. You can also access darknet version int github.
You can easily used train you net by following way:
python train.py --cfg yours.cfg --weights yours.weights --batch-size 8 --epoch 50 --devices 0,1

And you can test your trained model as this:
python detect.py --cfg yours.cfg --weights yours.weights --source img_path
