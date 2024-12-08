# Super Resolution API

基于 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 预训练模型 和 `onnxruntime` 的图像超分辨率 Web API, 支持 CPU 推理.

## 特性

- 硬件要求低, 可在 1C1G 服务器上运行. 可选的 GPU
- 基于任务ID的异步处理
- 可选的分布式部署, 各节点共同推理单张图像
- 高可配置性, 灵活的路由参数

## 运行模式

| 模式     | 说明                                                       |
| -------- | ---------------------------------------------------------- |
| `single` | 单机部署模式, 默认, 所有推理在单机上完成                   |
| `master` | 主节点模式, 接收并分发任务给从节点处理                     |
| `slave`  | 从节点模式, 接收主节点分发的任务, 也可作为单机节点接受请求 |

客户端无需关心节点类型, 所有模式下具有相同的 API


## 部署
### Docker (仅支持 CPU)

可使用构建好的 Docekr 镜像在 docker compose 中运行

```bash
wget https://raw.githubusercontent.com/krau/super-resolution-api/main/docker-compose.yml
wget https://raw.githubusercontent.com/krau/super-resolution-api/main/.env.example
mv .env.example .env
```

参考 [配置](#配置) 环境变量部分修改 `.env` 文件, 然后运行

```bash
docker compose up -d
```

### 源码运行
#### 环境

使用任意工具创建虚拟环境, Python==3.11 , 未测试其他版本

安装 opencv 相关依赖库, 在 ubuntu/debian 系统下可使用以下命令, 其他系统请自行安装

```bash
sudo apt update && apt install ffmpeg libsm6 libxext6 -y 
```

#### 安装依赖

```bash
git clone https://https://github.com/krau/super-resolution-api
cd super-resolution-api
pip install -r requirements.txt
```

默认使用 CPU 推理, 如需使用 GPU 推理, 请自行解决 CUDA 和 cuDNN 环境问题, 并安装 `onnxruntime-gpu`

```bash
pip install onnxruntime-gpu
```

然后将 `provider` 配置为 `CUDAExecutionProvider`

#### 运行

在运行之前, 参考 [配置](#配置) 部分进行配置, 然后运行

```bash
python main.py
```

## 配置

支持使用 `toml` 配置文件和环境变量进行配置, 优先级: 环境变量 > 配置文件

配置文件: `settings.toml`

```toml
# onnxruntime provider, 可选: "CPUExecutionProvider", "CUDAExecutionProvider"
provider = "CPUExecutionProvider"
# API Token, 客户端应在 X-Token 头中传递
token = "qwqowo"
# 临时文件夹
temp_dir = "./temp"
# 输出文件夹
output_dir = "./output"
# 日志级别
log_level = "DEBUG"
# Redis URL
redis_url = "redis://localhost:6379"
# 监听地址
host = "0.0.0.0"
# 监听端口
port = 39721
# 最大可接受的超时时间
max_timeout = 300
# 默认超时时间
timeout = 30
# 最大并行处理图像块数
max_thread = 8
# 运行模式, 可选: "master", "slave", "single"
mode = "single"

###########################
# 以下仅 master 模式下需配置 #
##########################
# 向从节点检查推理结果的间隔, 单位: 秒
worker_check_interval = 5
# 从节点信息过期时间, 单位: 秒
worker_expire = 120
##########################

##########################
# 以下仅 slave 模式下需配置 #
#########################
# 主节点地址
master_url = "http://localhost:39721"
# 主节点 Token
master_token = "qwqowo"
# 从节点 ID, 用于区分不同节点, 需全局唯一
worker_id = ""
# 向主节点注册的自身地址, 主节点将使用此地址联系从节点
worker_url = "http://localhost:39721"
# 向主节点注册的周期, 单位: 秒, 应小于主节点上的 worker_expire
register_interval = 30
##########################
```

环境变量: 以 `SRAPI_` 作为前缀, 解释见上

```bash
SRAPI_PROVIDER = "CPUExecutionProvider"
SRAPI_LOG_LEVEL = "INFO"
SRAPI_TOKEN = "token"
SRAPI_REDIS_URL = "redis://localhost:6379"
SRAPI_HOST = "0.0.0.0"
SRAPI_PORT = 39721
SRAPI_TEMP_DIR = "./temp"
SRAPI_OUTPUT_DIR = "./output"
SRAPI_MAX_TIMEOUT = 300
SRAPI_TIMEOUT = 30
SRAPI_MAX_THREAD = 8
SRAPI_MODE = "master"
SRAPI_WORKER_CHECK_INTERVAL = 5
SRAPI_WORKER_EXPIRE = 120
SRAPI_MASTER_URL = "http://localhost:39721"
SRAPI_MASTER_TOKEN = "token"
SRAPI_WORKER_ID = "special_id"
SRAPI_WORKER_URL = "http://localhost:39721"
SRAPI_REGISTER_INTERVAL = 30
```

## 路由

所有路由均需要使用 `X-Token` 头传递 API Token

### GET `/` : 返回 API 信息

response example:

```json
{
    "message": "Super Resolution API is running as single mode",
}
```

### POST `/sr` : 超分辨率

request:

| 参数         | 类型                | 说明                                                   |
| ------------ | ------------------- | ------------------------------------------------------ |
| `file`       | multipart/form-data | 待处理图像文件                                         |
| `tile_size`  | int                 | 图像块大小, 32<=tile_size<=128, 默认 64                |
| `scale`      | int                 | 放大倍数, 2<=scale<=8, 默认 4                          |
| `skip_alpha` | bool                | 是否跳过 alpha 通道, 默认 False                        |
| `resize_to`  | str                 | 缩放到指定大小,  两种格式: 1. 1920x1080 2. 1/2. 默认无 |
| `url`        | str                 | 图像下载 URL, 与 `file` 二选一                         |
| `timeout`    | int                 | 超时时间, 默认遵循 `timeout` 配置项                    |

response example:

```json
{
    "task_id": "133313131-1",
    "message": "Success",
}
```

### GET `/result/{task_id}` : 获取任务状态

response example:

success:

```json
{
    "result":{
        "status":"success",
        "path":"./output/133313131-1.png",
        "size": 102400,
    }
}
```

pending:

```json
{
    "result":{
        "status":"pending",
    }
}
```

### GET `/result/{task_id}/download` : 下载结果

当任务状态为 `success` 时, 返回文件流

---

## TODO

- [ ] 更合理的图片切割与任务分发策略
- [ ] 回调式通知主节点任务状态, 避免轮询
- [ ] 多模型支持

## Reference & Thanks

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [MoeSR](https://github.com/TeamMoeAI/MoeSR)
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer)
- all dependencies...