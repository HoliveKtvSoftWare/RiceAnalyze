from ultralytics import YOLO

# 训练
model = YOLO('./yolov8n-seg.pt')
# Train the model
results = model.train(data='/public/zhuxiaoying/shuidao/CodeAndDataset/code/ultralytics-main/ultralytics/models/yolo/segment/data.yaml', device='0')


#  预测
# model = YOLO('/public/wzxy_a40/pangguangyao/lwy/competition/111/ultralytics-main/ultralytics/models/yolo/segment/runs/segment/train5/weights/best.pt')
# result = model("/public/wzxy_a40/pangguangyao/lwy/competition/111/2/total/test/images/Y418-1.2.jpg",save=True)

