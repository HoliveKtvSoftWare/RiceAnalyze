# 剑叶分析

新分析使用 OUR 模型，权重放在 `models_yolo/leaf/leaf-our-best.pt`。
路径由 `.env` 中的 `LEAF_MODEL_PATH` 指定，相对路径固定从后端目录解析。
茎秆权重单独放在 `models_yolo/stem/yolov8s-1.pt`。

安装方法见 `DEPLOY.md`，OUR 分支与掩膜边界适配见 `REFINE_HEAD.md`。
两类模型共用官方 Ultralytics 环境，不需构建运行库。

历史的 exp59、exp08、exp12、exp61 和对比模型任务标识仍保留，
已有结果可以查看、筛选和导出；停用模型不出现在新分析选项中。
旧权重无需放到新服务器，历史结果导出读取的是保存的 JSON，不会重新运行旧模型。
