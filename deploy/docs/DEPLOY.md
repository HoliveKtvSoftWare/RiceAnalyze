# 安装和部署

现在只需一个普通 Python 环境，同时运行茎秆模型和剑叶 OUR 模型。

## 需要提供什么

1. 后端 `RiceAnylaze` 和前端 `RiceAnylazeWeb` 的当前代码，放在同一父文件夹。
2. 方案A部署资源包中的两个模型，解压到后端，目录结构如下：

```text
RiceAnylaze/
  app/                                  后端代码，包含 OUR 适配
  models_yolo/stem/yolov8s-1.pt           茎秆模型
  models_yolo/leaf/leaf-our-best.pt        剑叶模型
  requirements.txt                       pip 依赖清单
  .env.example                           配置模板
  deploy/runtime/                        启动脚本
RiceAnylazeWeb/                           前端代码
```

新部署不需要 `.ultra_refine`、`.pylibs`、定制 YOLO 源码或第二个 Python 环境。
历史模型仅支持查看与导出已有结果；新分析的选项只有茎秆和剑叶 OUR。

## 第一次安装

安装 Python 3.12（64位，已验证；依赖支持3.10–3.12）、Node.js 22.12以上，以及 PostgreSQL。
在后端文件夹打开终端，执行：

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

不要使用 `--system-site-packages`。普通 pip 安装即可，不必安装训练项目或修改 site-packages。
默认即可使用 CPU；有 NVIDIA GPU 时，可按 PyTorch 官方方法安装对应 CUDA 的
`torch 2.4.1` 和 `torchvision 0.19.1`。OUR 适配固定对应 `ultralytics 8.3.27`，不要单独升级。

打开 `.env`，只需要按实际情况修改：

- `DATABASE_URL`：PostgreSQL 用户、密码、服务器地址、端口和库名，保留 `+asyncpg`。
- `SECRET_KEY`：为这台服务生成一个新密钥：`python -c "import secrets; print(secrets.token_hex(32))"`。

先在 PostgreSQL 中建立连接串所指定的数据库并授权。首次启动会创建表，不会代建数据库。
如果迁移已有账号和记录，需要另外恢复 PostgreSQL 备份和 `app_storage` 文件夹。
全新部署可以使用空数据库，不需要复制原用户数据。

以下路径通常无需修改，均相对于后端文件夹，项目可以换盘符或挪位置：

```dotenv
STORAGE_PATH="./app_storage/user_data"
YOLO_MODEL_PATH="./models_yolo/stem/yolov8s-1.pt"
LEAF_MODEL_PATH="./models_yolo/leaf/leaf-our-best.pt"
```

`REDIS_URL` 只为兼容旧配置保留，当前后台任务不要求启动 Redis。
YOLO 推理和图割在一个服务进程内串行执行，避免共用模型实例造成交叉结果。
启动脚本使用一个 Uvicorn worker；不要用多 worker 重复占用模型内存。
从旧版本升级时，必须将 `LEAF_MODEL_PATH` 从旧 exp59 路径改为上面的 OUR 路径。
配置了不含 refinement 的旧权重时，新后端会拒绝执行并给出提示，避免把旧模型结果记成 OUR。
旧版 `.env` 的 `FORK_*` / `SIDECAR_SCRIPT` 键可以删除，新后端不读取它们。
不要把包含真实密码的 `.env` 提交到 Git。

## 构建前端并启动

本次更新的前端默认请求 `http://localhost:8000/api`，适用于本机验收。
部署给其他电脑访问时，建议在前端构建前创建 `.env.production`，使用同源代理：

```dotenv
VITE_API_BASE_URL=/api
```

然后执行：

```bat
npm ci
npm run build
```

回到前后端的父文件夹，执行：

```bat
RiceAnylaze\deploy\runtime\start-all.cmd
```

默认打开 `http://localhost:5173/`；后端接口文档为 `http://localhost:8000/docs`。
可使用 `RiceAnylaze\deploy\runtime\status.cmd` 检查端口与数据库连接。
启动脚本优先用 `RICE_PYTHON` 环境变量，其次用后端 `.venv/Scripts/python.exe`，
最后用当前 PATH 的 `python`。通常按上面的 `.venv` 安装即可。

如果 PostgreSQL 的 bin 目录不在 PATH，可以设置系统环境变量 `PGBIN`；
脚本也会尝试查找 Program Files 下的 PostgreSQL 安装目录。

## 日后更新

更新代码后，在后端执行 `.venv\Scripts\python.exe -m pip install -r requirements.txt`。
前端有更新时重新执行 `npm ci`、`npm run build`，然后重启服务。
模型权重被 Git 忽略，需从部署资源包补齐；用户数据和私有 `.env` 也不随 Git 分发。

## 剑叶算法选项

`.env` 默认 `LEAF_POSTPROCESS="graphcut"`：side1/side2使用方案 B 图割；body1/body2
使用局部边缘修正与受约束平滑。配置名保持兼容，不需要修改已有.env。
body沿原模型轮廓在有限范围内结合局部颜色与连续分数寻找边缘，孔洞多边形保持原样，
不可靠的局部移动会回退，并检查拓扑、面积、自交及最终主脉/空腔包含关系。
主脉与空腔可按原标注包含；不同组织的新增像素只归属一个区域，避免互相侵占。
side图割可根据原图修复误分的内部假孔洞；不添加单独的填孔步骤，也不提供全部填孔配置。
small/big 的原有标注保持独立覆盖。

`conservative` 为原有保守side修正，`off` 停用区域后处理。
这些配置只作用于之后的新分析；历史 JSON、人工修改的 JSON 和已有 Excel 不会自动改写。

性状计算采用独立版纯几何算法，自动寻找两端，输出实际长度；页面和导出暂不提供排列顺序，测长内部排序保留。
缺少有效图内比例尺时，物理量留空，维管束计数保留。
接口的 `scale` 表示图中标尺的物理长度（默认500µm），不是缺失标尺时的像素当量。
详细规则见 `../../docs/side-length.md`；OUR 适配说明见 `REFINE_HEAD.md`。
