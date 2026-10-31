# OUR 模型适配说明

当前推理使用官方 pip 包 `ultralytics==8.3.27`，无需旁路进程或自定义环境。
适配代码位于 `app/infrastructure/inference/our.py`，由 `app/services/yolo_inference.py` 缓存加载时启用。

OUR 权重中含有 `mask_refine` 子模块。官方类可以反序列化这些层，但它的
`Segment.forward` 不会调用这些层。因此，单纯使用官方 `YOLO(...)` 虽然不会报错，
却会明显改变掩膜。这里给这一个模型的检测头设置项目内的 `OurSegment` 类，
在原型掩膜后执行训练好的 refinement；不改官方安装目录，也不改全局 `Segment`。

后处理还保持原正确运行时的三项规则：

1. 去除 letterbox 留白时使用对称的 round(pad±0.1) 取整。
2. 保留小批量 CPU 掩膜裁框的取整与空掩膜过滤规则。
3. 小维管束的 `.xy` 轮廓使用所有区域合并；主区域仍由原拓扑转换保留多区域及孔洞。

这些适配仅用于含 refinement 的 OUR 权重。茎秆继续走官方原有检测头和后处理。
新分析类型为 `stem` 与 `leaf`，剑叶模型选项只有 OUR。历史 `leaf_our` 等任务 key
仍归入剑叶进行查看和导出，不改写已有记录的模型来源。旧页面传来的模型名称
归一到当前默认模型，新任务会记录实际使用的 OUR 或茎秆模型名称。
OUR 任务还要求权重含 `mask_refine`，旧配置误指向 exp59 等权重时会明确失败并提示修改路径。

body1、body2、side1、side2的连续掩膜分数随 OUR 推理结果保留，为side方案 B图割和body局部轮廓修正提供输入，避免把二值掩膜冒充概率。
性状计算在 `app/domain`，使用独立版自动端点与软引导测长。区域按最大净面积实例选择，
保留同一实例的多个连通部分、孔洞和浮点轮廓，不提供单独的全部填孔步骤。
更换权重或升级 Ultralytics/PyTorch 后，需要重新做真实图片的掩膜和性状对照验证。
