# OOPZ Voice Chat / OOPZ 语音聊天

让 AstrBot 加入 **OOPZ** 语音频道，实现 听→想→说 的语音对话。

## 功能

- 通过 [oopz-sdk](https://pypi.org/project/oopz-sdk/) 连接 OOPZ 平台
- 加入 / 离开 OOPZ 语音频道
- 被动监听 + 本地唤醒词 (`faster-whisper`) 检测
- 唤醒后调用 AstrBot STT → LLM → TTS Provider 完成语音回复
- TTS 音频推回语音频道
- WebUI Dashboard：可视化会话状态、操作按钮、SSE 实时推送
- 支持 AstrBot 人格系统

## 架构

```
OOPZ 平台
  ↕ (WebSocket)
oopz-sdk (OopzBot)
  ↕ (voice API)
本插件
  ├── 音频帧接收 → VAD → 唤醒词检测
  ├── STT (AstrBot Provider)
  ├── LLM (AstrBot Provider)
  └── TTS (AstrBot Provider) → 推回语音频道
```

## 前置依赖

### 1. 安装 Playwright Chromium

`oopz-sdk` 的语音推流依赖 Playwright Chromium：

```bash
python -m playwright install chromium
```

### 2. 插件依赖

AstrBot 会自动安装 `requirements.txt` 中的依赖（`aiohttp` / `pydub` / `numpy` / `faster-whisper`）。

> **注意**：`oopz-sdk` **不**在 `requirements.txt` 中（与 AstrBot 的 `cryptography` 版本冲突）。
> 插件启动时会自动用 `pip install oopz-sdk --no-deps` 安装。首次启用后需**重载插件**。

## 快速开始

### 1. 获取 OOPZ 凭据

登录 OOPZ 客户端后，从以下渠道提取：

| 凭据 | 说明 |
|------|------|
| `device_id` | 设备 ID（UUID） |
| `person_uid` | 用户 UID |
| `jwt_token` | 登录态 JWT |
| `private_key` | RSA 私钥（PEM 格式） |

获取方式：
- 浏览器开发者工具 → Network → 查找 `device_id`、`person_uid`、`jwt_token`
- RSA 私钥：使用 [Oopzbot 凭据工具](https://github.com/oopzbot/Oopzbot) 或从 IndexedDB 提取

### 2. 安装插件

将插件目录放到 AstrBot 的插件目录：
```
C:\Users\<用户名>\.astrbot\data\plugins\astrbot_plugin_oopz_voice\
```

### 3. 启用插件

1. 打开 AstrBot WebUI → 插件管理
2. 找到 **OOPZ Voice Chat** → 启用
3. 配置凭据：填入 `device_id`、`person_uid`、`jwt_token`、`private_key`
4. 配置 Provider：填入 `stt_provider_id`、`tts_provider_id`（必须是 AstrBot 中已启用的对应类型 Provider）
5. 重载插件（首次必做）

### 4. 加入语音频道

在 WebUI Dashboard 输入 `area_id` 和 `channel_id` 点击「加入」，或使用指令：
```
/oopz join <area_id> <channel_id>
```

## 配置项

### 凭据

| 字段 | 说明 |
|------|------|
| `auth.device_id` | OOPZ 设备 ID |
| `auth.person_uid` | OOPZ 用户 UID |
| `auth.jwt_token` | 登录态 JWT |
| `auth.private_key` | RSA 私钥 (PEM) |

### Provider

| 字段 | 说明 |
|------|------|
| `stt_provider_id` | 语音转文字 Provider ID（如 `mimo_stt`） |
| `tts_provider_id` | 文字转语音 Provider ID（如 `mimo_tts`） |
| `llm_provider_id` | 对话模型 Provider（留空使用当前会话模型） |

### 人格 & 对话

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `conversation.persona_id` | (空) | AstrBot 人格 ID，覆盖下方提示词 |
| `conversation.system_prompt` | 友好语音伙伴 | 自定义系统提示词 |
| `conversation.enable_history` | `true` | 启用 per-channel 对话历史 |
| `conversation.max_turns` | `12` | 保留的对话轮次 |

### 唤醒词

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `wake.wake_word` | `bot` | 主唤醒词 |
| `wake.wake_variants` | `["bot", "小机器人"]` | 备用唤醒词 |
| `wake.rms_gate` | `0.01` | VAD 音量门限 (0-0.1) |
| `wake.silence_ms_to_flush` | `700` | 静音结束录制 (ms) |
| `wake.max_listen_seconds` | `30` | 单次最长录音 (秒) |

### Whisper 本地识别

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `whisper.enabled` | `true` | 启用本地预检 |
| `whisper.model_size` | `tiny` | 模型大小 (`tiny`/`base`/`small`) |
| `whisper.device` | `cpu` | 推理设备 (`cpu`/`cuda`) |

### TTS 播放

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `tts_playback.speed` | `1.0` | 播放速度 (0.5-2.0) |
| `tts_playback.max_text_length` | `500` | 单次最大字符数 |
| `tts_playback.split_long_text` | `true` | 长文本按句切分 |

## 指令

```
/oopz status                          # 列出所有频道状态
/oopz join <area> <channel>           # 加入语音频道
/oopz leave [area] [channel]          # 离开
/oopz say <area> <channel> <text>     # 直接 TTS 播报
/oopz interrupt [area] [channel]      # 打断播放
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
     ▼
┌─────────┐
│  THINK  │ (LLM 生成回复)
└────┬────┘
     ▼
┌─────────┐
│  TTS    │ (云端 TTS → WAV)
└────┬────┘
     ▼
┌─────────┐
│  SPEAK  │ (推流到 OOPZ)
└────┬────┘
     │ 播放完毕 / 被打断
     └─→ IDLE
```

## WebUI Dashboard

访问 AstrBot WebUI → 插件 Pages → **OOPZ 语音控制台**：

- 连接状态实时显示（已连接/未连接）
- 加入/离开语音频道
- 活跃会话列表（状态、轮次、输入/输出）
- 直接播报 TTS
- Provider 配置查看
- 频道对话历史
- SSE 实时推送

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| `oopz-sdk` 导入失败 | `pip install oopz-sdk --no-deps`，然后重载插件 |
| 推流无声音 | `python -m playwright install chromium` |
| 唤醒词不触发 | 调整 `whisper.model_size` 为 `base`；调小 `wake.silence_ms_to_flush` |
| TTS 报错 | 确认 `tts_provider_id` 对应 Provider 已启用 |
| WS 频繁断线 | 凭据过期，重新提取 `jwt_token` 与 `private_key` |
| 人格不生效 | `persona_id` 必须与 AstrBot 人格管理中的名称完全一致 |

## 致谢

- [oopz-sdk](https://pypi.org/project/oopz-sdk/) — OOPZ Python SDK
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) — Bot 框架
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 本地语音识别
- [pydub](https://github.com/jiaaro/pydub) — 音频处理

## License

MIT
