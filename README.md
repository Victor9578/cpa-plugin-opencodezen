# cpa-plugin-opencodezen

<div align="center">

![logo](logo.svg)

**OpenCode Zen free-tier models as a native CLIProxyAPI provider**

[![Go](https://img.shields.io/badge/go-1.26-00ADD8?logo=go)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Victor9578/cpa-plugin-opencodezen?include_prereleases)](https://github.com/Victor9578/cpa-plugin-opencodezen/releases)

</div>

用于 [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)（v8.0.4+）的**原生执行器插件**，让你可以直接在 CLIProxyAPI 中无痛使用 [OpenCode Zen](https://opencode.ai/zen) 的免费系列模型（如 `mimo-v2.6-flash-free`、`muse-spark-1.3-contributor-free` 等），无需修改或重新编译 CPA 宿主二进制。

---

## 解决的问题

1. **端点路由自动分流（Endpoint Split）**：
   - OpenCode Zen 的官方文档（[端点表](https://opencode.ai/docs/zh-cn/zen/#端点)）列明每个模型由 `/responses`、`/chat/completions` 或非 OpenAI 方言（`/messages`、`/systemone`）提供服务；`/models` 接口本身不返回端点字段。
   - 插件内置官方端点表快照（按模型家族分类，2026-10-01）：`gpt-`/`grok-`/`muse-` 走 `/responses`，免费系（`mimo-`/`longcat-`/`deepseek-`/`glm-` 等）走 `/chat/completions`；claude/qwen3.7/gemini/jev 等非 OpenAI 方言模型不会被注册（选中也无法工作）。
   - 端点表只是提示：未知模型按名称推断；未显式配置端点时，请求遇 400/404/405 自动换另一方言重试并记住结果，配置里显式写的 `endpoint` 永远优先。

2. **免费层门禁自动伪装（FreeTier Gate）**：
   - Zen 服务端对未通过官方 CLI 发起的请求有严格的校验（检查 `ses_` / `msg_` 规范格式会话头、客户端特征头 `cli`、`bash`/`read` 工具集、强制 `stream: true` 等），缺一不可，否则返回 `403 FreeTierError` 并封禁 Key 约 15 分钟。
   - 插件全自动伪装为官方 CLI 规范，确保百分百顺利通行。

3. **SSE 流式传输重拼与保活过滤（Stream Reassembly）**：
   - 修复上游分片造成的跨包 SSE 截断问题，确保下游收到的每个 `data: {...}` 均为完整合法的 JSON 对象。
   - 自动过滤上游 `: keep-alive` 心跳注释帧，避免 LobeHub、NextChat 等客户端因非法 JSON 报错崩溃。
   - 不转发上游 `[DONE]` 终止符（宿主会在插件流关闭后自动补齐，避免重复）。

4. **Responses → Chat 双向格式转换（Format Conversion）**：
   - 插件向宿主声明 `chat-completions` 输出格式，宿主据此为 responses/claude/gemini 客户端做翻译。
   - 走 `/responses` 上游的模型（如 `muse-spark`）返回的是 Responses API 事件流，插件会将其实时转换为 `chat.completion.chunk` 帧（含 role 首帧、增量 content、带 usage 的收尾帧），非流式请求则折叠为标准 `chat.completion` 对象。
   - 修复此前 muse 系列模型流式返回 `empty_stream`、非流式返回无法解析载荷的问题。

5. **断流自动重连与续写（Stream Auto-Resume，0.8.0+）**：
   - Zen 免费层上游偶发「静默截断」：SSE 流在 HTTP 层正常结束（200/EOF）却从未发出终止事件（`finish_reason` / `response.completed` / `[DONE]`），旧版会当作正常完成关闭流，客户端把写了一半的回答当成最终结果——这正是「会话莫名断掉、任务停在半路」的根因。
   - 0.8.0 起：上游传输错误（`unexpected EOF` 等）、5xx 突发、流中途断开、无终止事件的提前 EOF、60s 无字节停滞，都会自动重连（默认 3 次，退避 1s/2s/4s），下游桥接全程保持打开并发送心跳。
   - 若已向下游发出部分回答，重连请求会自动附带「半截回答 + 续写指令」，模型从断点无缝续写，客户端完全无感；未发出任何内容时则原样重放请求。
   - 工具调用流中断时无法安全续写（下游已收到 tool_call 头），此时按错误收尾，由客户端侧重试。
   - 可通过 `plugins.configs.zen.stream-retries` 调整重试次数（0 关闭，上限 10，默认 3）。

6. **无感集成，原生提供商体验**：
   - 支持两种凭证方式：在 Web 管理中心插件配置页直接填 `api-keys`（自动落盘为标准凭证文件），或在凭证目录手动放置 `zen` 认证 JSON 文件。
   - 服务地址与 API Key 支持动态感知生效。

---

## 安装方法

从 [Releases](https://github.com/Victor9578/cpa-plugin-opencodezen/releases) 下载适用于你系统架构的编译包（例如 `zen_0.7.6_linux_amd64.zip`）。

解压后将 `zen.so`（或 `zen.dylib` / `zen.dll`）放入 CPA 的插件目录（文件名必须为 `zen.so` / `zen.dylib` / `zen.dll`）：

```bash
/CLIProxyAPI/plugins/linux/amd64/zen.so
```

> **版本要求**：CPA v8.0.4+（插件 0.7.2 起适配 CPA v8 的 `auth.parse` 载荷格式）。

---

## 使用指南

### 1. 启用插件（`config.yaml`）

```yaml
plugins:
  enabled: true
  dir: "plugins"
  configs:
    zen:
      enabled: true
      # 推荐：直接在插件配置里添加 OpenCode key（逗号分隔）
      # 0.7.3 起也接受分号或空白分隔，每个 key 自动落盘为一条标准凭证
      # api-keys: "oc_sk_xxx,oc_sk_yyy"
      # 可选：从 /models 自动发现的模型中排除指定模型（逗号分隔，大小写不敏感）
      # exclude-models: "jev-1.13-free,space-bunny-free"
```

> `api-keys` 和 `exclude-models` 也可以在 CPA Web 管理中心的插件配置页直接填写。
> **注意**：多个 key 之间用逗号（`,`）分隔。0.7.3 之前分号（`;`）分隔会被当成一个拼接的长 key，导致上游 401。

### 2. 添加凭证（二选一）

**方式 A（推荐）：插件配置 `api-keys`**

在管理中心的插件配置页（或 `config.yaml` 的 `plugins.configs.zen.api-keys`）填入 OpenCode key。插件会自动通过宿主 `host.auth.save` 回调把每个 key 落盘为标准凭证文件（`zen-<hash>.json`），成为可调度、可冷却、可轮换的原生凭证。

**方式 B：认证目录手动放置文件**

在 CPA 的 `auth-dir`（例如 `/root/.cli-proxy-api/`）目录下创建一个 JSON 文件（例如 `zen-key.json`）：

```json
{
  "type": "zen",
  "provider": "zen",
  "api_key": "sk-your-zen-api-key-here",
  "base_url": "https://opencode.ai/zen/v1"
}
```

> **说明**：
> - 插件默认已注册以下免费模型：`mimo-v2.6-flash-free`、`mimo-v2.5-free`、`ling-3.0-flash-fin-free`、`nemotron-3-ultra-free`、`muse-spark-1.3-contributor-free`。
> - 插件会自动从 zen `/models` 接口发现新的免费模型（`*-free` 后缀，10 分钟缓存）并注册，无需重新编译；不想用的模型可用 `exclude-models` 排除。
> - 也可以在凭证 JSON 文件中添加 `"models": [{"name": "新模型名"}]` 手动指定。
> - 端点（chat / responses）自动识别：优先查内置官方端点表（文档快照 2026-10-01），未知模型按名称推断；未显式配置 `endpoint` 时，请求遇 400/404/405 自动换另一方言重试并记住。

### 3. 模型能力元数据（上下文窗口等）

插件会向 CPA 宣告模型能力元数据（上下文窗口、输出上限、输入模态、推理档位、显示名），下游客户端（pi、Codex 等）通过 `/v1/models` 读取 `context_window` / `max_tokens` 等字段。

内置规格表与 OpenCode 官方客户端自己的模型目录（models.dev 的 `opencode` provider）对齐，快照 2026-10-01：

| 模型 | 上下文 | 最大输出 | 推理档位 | 模态 |
|---|---|---|---|---|
| mimo-v2.6-flash-free / mimo-v2.5-free | 200K | 32K | 开关式 | 文本+图像 |
| ling-3.0-flash-fin-free | 256K | 32K | 开关式 | 文本 |
| nemotron-3-ultra-free | 1M | 128K | 开关式 | 文本 |
| nemotron-3.5-lightning-free | 256K | 256K | 开关式 | 文本 |
| muse-spark-1.3/1.2-contributor-free | 1M | 128K | minimal…xhigh | 文本+图像 |
| longcat-2.5-preview-free | 1M | 128K | 开关式 | 文本+图像 |
| space-bunny-free | 1M | 512K | low…max | 文本+图像 |
| big-pickle | 200K | 32K | 开关式 | 文本 |
| deepseek-v4-flash-free | 200K | 128K | low/high/max | 文本 |

> 注：不宣告能力时，CPA 的 codex 模型目录会用 gpt-5.5 模板兜底（272K/16K）——之前 muse 显示 272K 就是这个原因。

如需覆盖或为其他模型声明能力，在 `config.yaml` 的 `plugins.configs.zen.models` 中配置：

```yaml
plugins:
  enabled: true
  dir: "plugins"
  configs:
    zen:
      enabled: true
      models:
        - model: nemotron-3-ultra-free
          context-length: 1000000        # 上下文窗口（token 数）
          max-completion-tokens: 32768   # 最大输出 token 数
          input-modalities:              # 输入模态：text / image
            - text
          reasoning-levels: none, low, medium, high  # 推理档位
```

### 4. 代理配置（服务器 IP 被 Zen 风控时）

Zen 免费层有 IP 风控：同一服务器 IP 高频请求后会返回 `429 FreeUsageLimitError`（Rate limit exceeded）。此时可让 zen 流量走代理：

```yaml
# config.yaml 全局代理（插件上游请求经宿主 host.http.do 桥，会使用此设置）
proxy-url: "https://your-proxy:443"
```

> **重要**：
> - 插件上游请求走宿主的全局 `proxy-url`；**不支持**在 zen 凭证文件里单独设置 `proxy_url`（宿主对插件 HTTP 桥不传递 per-auth 代理）。
> - 全局代理会影响所有提供商。如果其他提供商（如国内天翼云）需要直连，可在 `openai-compatibility` 的 `api-key-entries` 里为该条目设置 `proxy-url: direct` 强制直连。
> - 代理需支持 HTTPS CONNECT（如 Clash/mihomo 的 http 节点 `https://host:443`）。Docker 部署时代理地址不能写 `127.0.0.1`（容器内不可达宿主回环），应写宿主 IP 或 `host.docker.internal`。
> - 免费代理节点不稳定，可能造成流式响应中断，建议选可靠节点。

### 5. 客户端接入

- **OpenAI 兼容客户端（LobeHub / NextChat 等）**：Base URL 填 `http://your-server:8080/v1`，Key 填 CPA 的 api-key。
- **pi**：安装 [`@router-for-me/pi-cliproxyapi-provider`](https://github.com/router-for-me/pi-cliproxyapi-provider) 包，配置 `baseUrl` 指向 CPA 即可；pi 走 `/backend-api/codex/responses`（Responses 协议），插件输出已验证可正常以 `response.completed` 收尾。

---

## 验证与测试

```bash
# 重启 CPA
docker restart cli-proxy-api

# 测试聊天（OpenAI 协议）
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $CPA_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mimo-v2.6-flash-free","stream":true,"messages":[{"role":"user","content":"Hello"}]}'

# 测试 Responses 协议（pi / Codex 客户端路径）
curl -s -N http://localhost:8080/backend-api/codex/responses \
  -H "Authorization: Bearer $CPA_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mimo-v2.6-flash-free","input":[{"role":"user","content":"Hello"}],"stream":true}'
```

---

## 故障排查

| 现象 | 原因与解决 |
|---|---|
| agent 客户端（pi 等）模型反复执行同一条命令、读不存在的文件 | 插件 ≤0.7.3 在 Responses→chat 转换时丢弃了 `function_call` / `function_call_output` 历史项，模型每轮看不到自己之前的工具调用与结果。升级到 **0.7.4+**。 |
| muse 等模型第二轮起必报 400，随后整轮 `auth_unavailable: no auth available` 中断 | 插件 ≤0.7.5 之前的 responses→responses 直通路径没有剥离重放的 `reasoning` 项（`rs_...` id 引用上游不保存的会话状态，报 `Referenced reasoning item was not found or has expired`），连续 400 还会让 CPA 把凭证标记不可用。升级到 **0.7.5+**（自动剥离 `reasoning` 项与 `previous_response_id`）。 |
| 选择 claude / gemini / jev 等模型后报 `dialect this plugin cannot speak` | 这些模型上游是 Anthropic Messages（`claude-*`、`qwen3.7-*` 等）、`/systemone`（`jev-*`）或 AI-SDK（`gemini-*`）端点，本插件只讲 chat-completions 与 /responses 两种方言；0.7.5 起这类模型直接不注册。确需强行指定时可在 `plugins.configs.zen models` 里显式写 `endpoint`（逃生口，通常无济于事）。 |
| 上游 `400 unknown_parameter` / `strict` 校验失败 | 插件 ≤0.7.4 的方言转换不彻底，`stream_options`、`store`、`max_tokens`↔`max_output_tokens`、`tool_choice` 形状等单侧参数会原样泄漏到另一方言，严格校验的 openai 兼容上游直接拒绝。升级到 **0.7.5+**。 |
| 上游因 max tokens 截断后，流式整轮报错、或非流式把截断当正常完成 | 插件 ≤0.7.4 把 `response.incomplete` 当失败抛错（流式），或折叠时丢掉 incomplete 状态导致 `finish_reason` 为 `stop`（非流式）。升级到 **0.7.5+**：按原因映射为 `finish_reason: length`/`content_filter` 并保留已生成内容与 usage。 |
| muse 模型工具调用无响应 / 空回复 | 插件 ≤0.7.3 的流转换器丢弃了 `function_call` 事件，agent 客户端收不到工具调用。升级到 **0.7.4+**。 |
| `auth_not_found: no auth available (providers=zen...)` | 插件 ≤0.7.1 在 CPA v8 上无法识别 `auth.parse` 的 `RawJSON` 载荷。升级到 **0.7.2+**；或手动在 auth 目录放凭证文件。 |
| 上游 `401` / 流中断（`upstream stream closed before a terminal event`） | `api-keys` 里多个 key 用了分号等非逗号分隔符，被拼成一个无效 key。升级 **0.7.3+**（支持逗号/分号/空白分隔），并删除已生成的坏凭证文件（管理中心凭证页可删）。 |
| `429 FreeUsageLimitError: Rate limit exceeded` | Zen 对服务器 IP 的风控。配置全局 `proxy-url` 走代理（见上文代理章节），或等待冷却后重试。 |
| `dial tcp 127.0.0.1:40000: connect: connection refused` | Docker 容器内访问不到宿主回环地址的代理。改用宿主 IP / `host.docker.internal`，或把代理跑在容器网络内。 |
| 管理中心连续输错密码后所有请求 403 | CPA 的 IP 封禁机制（`IP banned due to too many failed attempts`），等待解封或换 IP 访问管理接口。 |

---

## 版本历史

- **0.7.5**：端点路由对齐官方文档 + 跨方言转发加固 + 官方客户端行为对齐：
  - **修复 muse 第二轮起必 400 → auth_unavailable 中断**：responses→responses 直通路径现在剥离重放的 `reasoning` 项（`rs_` id 引用上游不保留的状态）与 `previous_response_id`；此前 pi 等 codex 方言客户端每轮重放 reasoning 项都会 400，连续失败后 CPA 把凭证标记不可用。
  - 端点：内置官方 Zen 端点表快照（按模型家族分类，抗文档变动；`/models` 无端点字段，文档是唯一机器可用来源）。注册与路由优先查表；非 OpenAI 方言模型（claude/gemini/jev/qwen3.7 等）不再被宣告；未显式配置端点时 400/404/405 **双向**换方言重试（此前仅 chat→responses 单向）并记住结果；显式 `endpoint` 配置永远优先。
  - 模型识别：内置规格表与 OpenCode 官方客户端自己的模型目录（models.dev `opencode` provider）对齐——上下文/最大输出/推理档位/模态/显示名。修正 mimo（1M→200K）、补齐 muse（1M/128K，修复客户端显示 272K 的问题）等全部免费模型。
  - 转发封装（对齐官方客户端）：补齐官方 6 大核心工具（bash/edit/glob/grep/read/write，官方 schema）并按字母序排序，无工具请求（压缩/总结）注入全套并设 `tool_choice: none`；`reasoning_effort` 按模型档位规约（muse: max→xhigh；开关式模型如 mimo 直接删除；未知模型钳到 low/medium/high），彻底消除 400 `Invalid request parameters`；删除非法顶层 `thinking` 对象；输出 token 按模型上限钳位。
  - 请求卫生：双向方言转换补齐参数映射与单侧字段清理 —— `max_output_tokens`↔`max_tokens`、`tool_choice` 函数对象互转、`response_format` json_schema 摊平、删除 `stream_options`/`logit_bias`/`store`/`include`/`truncation` 等单侧字段、`developer`→`system`、`input_image`→`image_url`、legacy `functions`/`function_call` 迁移、工具定义 `strict:null` 双向清理、assistant 文本与 `tool_calls` 并存时不再丢失。
  - 响应卫生：`response.incomplete` 不再当失败（流式）、不再被折叠成正常完成（非流式），按原因映射 `finish_reason: length|content_filter` 并保留已生成内容与 usage；chat 折叠保留上游真实 tool_call id（此前伪造 `call_%d`），上游不发 role 时补 `assistant`（此前空 role 会被宿主翻译层丢弃，表现为“空回复”）；`custom_tool_call` 事件全链路支持。
  - 自愈：上游 401/403（积分/鉴权/风控）时轮换 API key + 铸造全新官方格式会话 ID 重试一次；请求头补齐官方 `x-session-affinity` / `X-Session-Id`。
- **0.7.4**：修复 agent 客户端（pi / Codex 等 Responses 协议）多轮工具调用历史丢失的严重问题：
  - 请求方向：`function_call` / `function_call_output` 历史项在转换为 chat 格式时被静默丢弃，导致上游模型每轮“失忆”、重复执行相同工具调用（agent 死循环）。
  - 反向（chat 客户端 → responses 模型）：assistant 的 `tool_calls` 与 `role:"tool"` 结果消息同样丢失，现已转换为规范的 `function_call` / `function_call_output` 项。
  - 响应方向：muse 等 /responses 端点模型的 `function_call` 事件现在会转换为 chat `tool_calls` 增量帧（流式）与 `message.tool_calls`（非流式），agent 客户端终于能看到 muse 的工具调用。
  - 流空闲心跳：上游静默期间每 10 秒向下游发送 no-op `chat.completion.chunk`，防止 muse 等慢模型长思考时连接被中间层（NAT/代理/客户端传输层）掐断。
- **0.7.3**：`api-keys` / `exclude-models` 支持逗号、分号、空白多种分隔符（修复管理面板填分号导致 401 的问题）。
- **0.7.2**：适配 CPA v8 `auth.parse` 的 `RawJSON` 载荷格式；配置的 `api-keys` 自动落盘为标准凭证文件（`host.auth.save`）。
- **0.7.1**：管理面板新增 `api-keys`、`exclude-models` 配置字段。
- **0.7.0**：多凭证 auth 文件、模型实时发现、端点自动探测。
- **0.6.x**：Responses → Chat 流式格式转换、SSE 帧规范化修复。

---

## 致谢与参考

- [cpa-plugin-opencode-session-mapper](https://github.com/ahoo/cpa-plugin-opencode-session-mapper) - 会话头映射插件的先行者，为会话 ID 规范化处理提供了思路。
- [OpenCode2API](https://github.com/TiaraBasori/OpenCode2API) - 早期的 OpenCode 接入探索，为 Zen 免费层伪装提供了宝贵参考。
- 项目已在 [LINUX DO](https://linux.do/)社区发布并认可，欢迎社区的朋友来试用与反馈。

---

## License

[MIT](LICENSE)
