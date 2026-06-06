# OOPZ Voice Chat / OOPZ 语音聊天

让 AstrBot 加入 **OOPZ** 语音频道，与用户进行语音对话。

> 插件**不**桥接 OOPZ 文字消息到 AstrBot；只专注语音频道的 听→想→说 循环。

## 功能

- 通过 `oopz-sdk` 连接 OOPZ 平台
- 加入 / 离开 OOPZ 语音频道
- 被动监听 + 本地唤醒词 (`faster-whisper tiny`) 检测
- 唤醒后调用 AstrBot STT / LLM / TTS Provider 完成回复
- TTS 音频推回语音频道
- WebUI Dashboard：可视化会话状态、操作按钮、SSE 实时推送

## 前置依赖

### 1. 安装 OOPZ 浏览器后端

`oopz-sdk` 的语音推流依赖 Playwright Chromium：

```bash
python -m playwright install chromium
```

### 2. 插件依赖

AstrBot 会自动安装 `requirements.txt` 中的大部分依赖（`aiohttp` / `pydub` / `numpy` / `faster-whisper`）。

> **注意**：`oopz-sdk` **不**在 `requirements.txt` 中。
>
> 原因：`oopz-sdk 0.13.x` 把 `cryptography` 锁在 `<48`，但 AstrBot 核心已固定 `cryptography==48.0.0`。AstrBot 的 pip 安装器有版本保护，会拒绝降级。
>
> 解决方案：插件启动时**自动**用 `pip install oopz-sdk --no-deps` 旁路安装（`cryptography` 48.x API 向后兼容 oopz-sdk 用的 RSA 签名）。首次启动会执行该安装，然后**点一次 WebUI 的「重载插件」**即可让新模块生效。
>
> 如果自动安装失败（例如网络问题），可手动执行：
> ```bash
> pip install oopz-sdk --no-deps
> ```
> 然后重载插件。

## 快速开始（操作步骤）

### 1. 获取 OOPZ 凭据
登录 OOPZ 客户端后，从以下渠道提取：
- `device_id`、`person_uid`：网络请求中的 Header 或本地存储
- `jwt_token`：登录态 Token
- `private_key`：RSA 私钥（PEM 格式，含 `-----BEGIN PRIVATE KEY-----`）

### 2. 安装插件
将插件目录放到 AstrBot 的插件目录：
```
C:\Users\用户名\.astrbot\data\plugins\astrbot_plugin_oopz_voice\
```

### 3. 安装 Playwright Chromium（必须）
```bash
# 在 AstrBot 的 Python 环境中执行
python -m playwright install chromium
```

### 4. 启用插件并配置
1. 打开 AstrBot WebUI → 插件管理
2. 找到 **OOPZ Voice Chat** → 点击「启用」
3. 点击「配置」填入上述四个凭据
4. 填入 `stt_provider_id`、`tts_provider_id`（需在 AstrBot 已启用的 Provider 中选择）
5. 可选：填入 `llm_provider_id`（留空则使用当前会话的模型）
6. 可选：在 `auto_join_channels` 添加 `area_id:channel_id` 实现启动自动加入

### 5. 重载插件（首次必做）
首次启用后，插件会自动下载 `oopz-sdk`。完成后**必须在 WebUI 点击「重载插件」**让新模块生效。

### 6. 打开 Dashboard
WebUI → 插件 Pages → **OOPZ Voice Dashboard**：
- 输入 `area_id`、`channel_id` 点击「加入」
- 状态变为「监听中」后，在频道说话触发唤醒词（默认「bot」/「小机器人」）
- Bot 会自动 STT → LLM → TTS → 回复语音

## 配置详情

在 AstrBot WebUI → 插件管理 → OOPZ Voice Chat → 配置 中填写：

### 凭据

| 字段 | 说明 |
|---|---|
| `auth.device_id` | OOPZ 设备 ID |
| `auth.person_uid` | OOPZ 用户 UID |
| `auth.jwt_token` | 登录态 JWT |
| `auth.private_key` | RSA 私钥 (PEM 格式) |

获取方式：登录 OOPZ 客户端后从网络请求或本地存储提取。

### Provider

| 字段 | 说明 |
|---|---|
| `stt_provider_id` | 语音转文字 Provider **ID**（如 `mimo_stt`）。必须是在 AstrBot 中已启用的 STT 类型 Provider，**非 LLM 模型** |
| `tts_provider_id` | 文字转语音 Provider **ID**（如 `mimo_tts`）。必须是在 AstrBot 中已启用的 TTS 类型 Provider，**非 LLM 模型** |
| `llm_provider_id` | 对话模型 Provider（留空使用当前 AstrBot 会话绑定的 LLM） |

> 在 Dashboard 的「Provider 配置」面板可以查看当前已启用的 STT / TTS / LLM Provider 列表。

### 人格 & 对话

| 字段 | 说明 | 默认值 |
|---|---|---|
| `conversation.persona_id` | AstrBot 人格 ID。填写后使用该人格的提示词，覆盖下方自定义提示词 | (空) |
| `conversation.system_prompt` | 自定义系统提示词（未指定人格时生效） | 友好语音伙伴 |
| `conversation.enable_history` | 启用 per-channel 对话历史 | `true` |
| `conversation.max_turns` | 保留的对话轮次 | `12` |

> 语音对话会自动走 AstrBot 的消息管道，在 WebUI「会话管理」中可查看完整对话记录。

### 启动时自动加入

`auto_join_channels` 列表，每项格式 `area_id:channel_id`，例如 `123456:789012`。

### 唤醒词配置

| 字段 | 说明 | 默认值 |
|---|---|---|
| `wake.wake_word` | 主唤醒词 | `bot` |
| `wake.wake_variants` | 备用唤醒词列表 | `["bot", "小机器人"]` |
| `wake.rms_gate` | VAD 音量门限（0-0.1） | `0.01` |
| `wake.silence_ms_to_flush` | 静音多久结束录音(ms) | `700` |
| `wake.max_listen_seconds` | 单次最长录音(秒) | `30` |
| `wake.min_listen_ms` | 最小有效录音(ms) | `400` |

### Whisper 本地识别

| 字段 | 说明 | 默认值 |
|---|---|---|
| `whisper.model_size` | 模型大小（`tiny`/`base`/`small`） | `tiny` |
| `whisper.device` | 推理设备（`cpu`/`cuda`） | `cpu` |
| `whisper.compute_type` | 量化类型（`int8`/`float16`） | `int8` |
| `whisper.language` | 识别语言（`auto`/`zh`/`en`） | `auto` |

## 指令

```
# ... rest
```

### 2. 插件依赖

AstrBot 会自动安装 `requirements.txt` 中的大部分依赖（`aiohttp` / `pydub` / `numpy` / `faster-whisper`）。

> **注意**：`oopz-sdk` **不**在 `requirements.txt` 中。
>
> 原因：`oopz-sdk 0.13.x` 把 `cryptography` 锁在 `<48`，但 AstrBot 核心已固定 `cryptography==48.0.0`。AstrBot 的 pip 安装器有版本保护，会拒绝降级。
>
> 解决方案：插件启动时**自动**用 `pip install oopz-sdk --no-deps` 旁路安装（`cryptography` 48.x API 向后兼容 oopz-sdk 用的 RSA 签名）。首次启动会执行该安装，然后**点一次 WebUI 的「重载插件」**即可让新模块生效。
>
> 如果自动安装失败（例如网络问题），可手动执行：
> ```bash
> pip install oopz-sdk --no-deps
> ```
> 然后重载插件。

## 配置

在 AstrBot WebUI → 插件管理 → OOPZ Voice Chat → 配置 中填写：

### 凭据

| 字段 | 说明 |
|---|---|
| `auth.device_id` | OOPZ 设备 ID |
| `auth.person_uid` | OOPZ 用户 UID |
| `auth.jwt_token` | 登录态 JWT |
| `auth.private_key` | RSA 私钥 (PEM 格式) |

获取方式：登录 OOPZ 客户端后从网络请求或本地存储提取。

### Provider

| 字段 | 说明 |
|---|---|
| `stt_provider_id` | 语音转文字 Provider |
| `tts_provider_id` | 文字转语音 Provider |
| `llm_provider_id` | 对话模型 Provider（留空使用当前 AstrBot 会话） |

### 启动时自动加入

`auto_join_channels` 列表，每项格式 `area_id:channel_id`，例如 `123456:789012`。

## 指令

```
/oopz status                          # 列出所有频道状态
/oopz join <area> <channel>           # 加入语音频道
/oopz leave [area] [channel]          # 离开
/oopz say <area> <channel> <text>     # 不走 STT，直接 TTS
/oopz interrupt [area] [channel]       # 打断播放
/oopz set wake <word>                 # 改唤醒词
/oopz set tts <provider_id>           # 切换 TTS Provider
/oopz set stt <provider_id>           # 切换 STT Provider
/oopz set llm <provider_id>           # 切换 LLM Provider
/oopz history clear [area] [channel]  # 清空某频道历史
```

## 工作流程

```
┌─────────┐
│  IDLE   │ ◀──── timeout / leave
└────┬────┘
     │ 收到 PCM 帧
     ▼
┌─────────┐
│ LISTEN  │ (VAD 累积 + 本地唤醒词判定)
└────┬────┘
     │ 唤醒命中
     ▼
┌─────────┐
│  STT    │ (云端 STT 完整识别)
└────┬────┘
     │
     ▼
┌─────────┐
│  THINK  │ (LLM)
└────┬────┘
     │
     ▼
┌─────────┐
│  TTS    │ (云端 TTS → wav)
└────┬────┘
     │
     ▼
┌─────────┐
│  SPEAK  │ (PCM 推流到 OOPZ)
└────┬────┘
     │ 播放完毕 / 被打断
     └─→ IDLE
```

## WebUI

访问 WebUI → 插件 Pages → **OOPZ 语音控制台**：

- 实时显示各频道状态（空闲 / 监听中 / 识别中 / 思考中 / 合成中 / 播放中）
- 强制加入 / 离开 / 打断
- 直接播报文字 TTS
- **Provider 配置面板**：显示当前已启用的 STT / TTS / LLM Provider，可点击刷新
- **频道对话历史**：选择频道后可加载完整的 User/Assistant 对话记录
- 人格 & 提示词显示
- SSE 实时推送状态变更

> 语音对话会自动写入 AstrBot 的会话管理系统，在 WebUI「会话管理」中也能看到对话记录。

## 故障排查

- **`oopz-sdk` 导入失败** → `pip install oopz-sdk --no-deps`
- **推流无声音** → `python -m playwright install chromium`
- **STT/TTS 下拉只显示 LLM 模型** → STT 和 TTS Provider 需在 AstrBot 提供商管理中添加指定类型的 Provider。在 OOPZ 控制台的 Provider 面板可以查看可用列表
- **本地唤醒词不触发** → 在 `whisper.model_size` 中调整为 `base`；调小 `wake.silence_ms_to_flush`
- **TTS 报错** → 确认 `tts_provider_id` 对应 Provider 已启用，且为 `text_to_speech` 类型
- **OOPZ WS 频繁断线** → 凭据过期，重新提取 `jwt_token` 与 `private_key`
- **静音段被误切成碎片** → 调大 `wake.rms_gate`（默认 `0.01`），或调大 `wake.silence_ms_to_flush`（默认 `700`）
- **人格不生效** → `persona_id` 必须与 AstrBot 人格管理中的名称完全一致。填写后保存配置并重载插件。

## 致谢

- [oopz-sdk](https://pypi.org/project/oopz-sdk/) — 社区维护的 OOPZ Python SDK
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) — 强大的 Bot 框架
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 本地语音识别
- [pydub](https://github.com/jiaaro/pydub) — 跨平台静音检测（同时使用 [ffmpeg](https://ffmpeg.org/) 做后端格式转换）
- [webrtcvad](https://github.com/wiseman/py-webrtcvad) — 可选的高精度 VAD（Windows Python 3.12 用户需要 C 工具链才能从源码编译）

## License

MIT
