from ultralytics import YOLO

# 训练
model = YOLO('/public/zhuxiaoying/shuidao/CodeAndDataset/code/ultralytics-main/ultralytics/cfg/models/v8/yolov8x-ema.yaml').load('./yolov8x-seg.pt')
# Train the model
results = model.train(data='/public/zhuxiaoying/shuidao/code_excute/data.yaml', task='segment', device='0')


#  预测
# model = YOLO('/public/zhuxiaoying/shuidao/code_excute/runs/segment/train8/weights/best.pt')
# result = model("/public/zhuxiaoying/shuidao/shuidao_new/test/images/21c-03.jpg",save=True)
# command > output.txt

