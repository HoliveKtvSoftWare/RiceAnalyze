# RiceAnylaze 后端

当前后端使用方案A：一个普通 Python 环境运行茎秆分割与剑叶 OUR 分割。
OUR 的适配代码在 `app/infrastructure/inference/our.py`，安装官方依赖即可运行。

## 首次部署

准备 Python 3.12、PostgreSQL，以及部署资源包中的两个模型：

```text
models_yolo/stem/yolov8s-1.pt      茎秆模型
models_yolo/leaf/leaf-our-best.pt  剑叶 OUR 模型
```

在后端目录执行：

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

修改 `.env` 中的 `DATABASE_URL` 和 `SECRET_KEY`，并在 PostgreSQL 中建立对应数据库。
模型和存储路径默认相对于后端目录，无需填当前电脑的盘符。
如果已有 `.env`，沿用其中的数据库和密钥，并确认 `LEAF_MODEL_PATH` 指向 `leaf-our-best.pt`。

启动后端：

```bat
deploy\runtime\start-backend.cmd
```

接口文档：<http://localhost:8000/docs>。

前端 `RiceAnylazeWeb` 与本目录放在同一父目录。
在前端执行 `npm ci`、`npm run build` 后，可以使用 `deploy\runtime\start-all.cmd` 启动前后端。
页面地址：<http://localhost:5173/>。

详细步骤见 [安装和部署](deploy/docs/DEPLOY.md)。

## 代码放在哪里

| 目录 | 作用 |
|---|---|
| `app/api` | HTTP 接口、登录和权限 |
| `app/api/endpoints` | 上传、批次、历史、指标、Excel 和 JSON 接口 |
| `app/services` | 后台分析、模型缓存、指标计算与导出 |
| `app/core` | 配置、路径解析、历史任务类型兼容 |
| `app/domain` | 性状指标和侧脉测长算法 |
| `app/infrastructure/inference` | 官方 YOLO 推理、OUR 适配及结果绘制 |
| `app/database`、`app/models` | 数据库连接、表结构升级和数据模型 |
| `deploy/runtime` | 启动与状态检查 |

新分析只提供茎秆和剑叶 OUR。旧模型标识保留用于查看和导出历史结果。
侧脉长度自动寻找两端，使用独立版软引导路径与留空规则；页面和导出暂不提供排列顺序，
测长内部的维管束排序保留。不再导出周长÷2和测长状态。算法口径见 [侧脉测长说明](docs/side-length.md)。

剑叶默认对 `side1`、`side2` 使用方案 B 原图图割，对 `body1`、`body2` 使用局部边缘修正与受约束平滑。
body沿模型原有外轮廓在有限距离内寻找原图边缘，并结合连续分数与局部颜色判断；孔洞多边形保持原样。
可靠性不足的局部保留原轮廓，最终检查面积、拓扑、位移和主脉与空腔的包含关系。
small、big 是独立覆盖标注，不提供额外填孔步骤或开关。
默认配置仍为 `LEAF_POSTPROCESS=graphcut`，这个兼容配置名启用上述混合流程。配置只影响之后的新分析，已有 JSON 不会被改写。

## Git 包含什么

Git 包含源码、依赖清单、配置模板、启动脚本和使用文档。
模型权重、私有 `.env`、用户数据、Python 环境、运行缓存、离线导出任务和本地实施资料均被忽略。
后端回归测试和验收报告单独归档，不放入部署源码。
同事拉取代码后，还需要补齐两个模型和自己的 `.env`，再安装依赖。
