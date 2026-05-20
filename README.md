# Research Agent

一个基于 LangGraph 的 Research Agent。它可以根据用户输入决定是否进行网页搜索和网页读取，生成可保存的 Research Markdown 笔记，并支持每天定时搜索新闻、保存笔记、发送 Outlook 邮件。

## Quick Start

### 1. 准备 Python 环境

项目使用 Python 3.12。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

安装当前代码需要的依赖：

```powershell
python -m pip install python-dotenv langchain-core langgraph langchain-deepseek tavily-python httpx requests mcp msal pydantic
```

### 2. 创建 `.env`

在项目根目录创建 `.env`。不要把 `.env` 提交到 Git。

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key

MS_CLIENT_ID=your_microsoft_app_client_id
MS_TENANT_ID=consumers
MAIL_WHITELIST=your_email@example.com

DAILY_EMAIL_TO=your_email@example.com
DAILY_EMAIL_QUERY=今日新闻
DAILY_EMAIL_HOUR=8
DAILY_EMAIL_MINUTE=0
DAILY_EMAIL_CHECK_INTERVAL_SECONDS=30
DAILY_EMAIL_RETRY_COUNT=2
DAILY_EMAIL_RETRY_DELAY_SECONDS=10
DAILY_EMAIL_TIMEOUT_SECONDS=300
```

说明：

- `DEEPSEEK_API_KEY`：LangGraph 中的 DeepSeek 模型调用密钥。
- `TAVILY_API_KEY`：Tavily 搜索密钥，也兼容 `TVLY_API_KEY`。
- `MS_CLIENT_ID`：Microsoft 应用的 client id，用于 Outlook 设备码登录。
- `MS_TENANT_ID`：个人 Outlook 账号通常用 `consumers`。
- `MAIL_WHITELIST`：允许收发邮件的白名单邮箱，多个邮箱用英文逗号分隔。
- `DAILY_EMAIL_TO`：每日简报发送到哪个邮箱，必须在白名单内。
- `DAILY_EMAIL_RETRY_COUNT`：每日任务失败后额外重试几次，默认 `2`。
- `DAILY_EMAIL_RETRY_DELAY_SECONDS`：两次重试之间等待几秒，默认 `10`。
- `DAILY_EMAIL_TIMEOUT_SECONDS`：单次每日任务最多运行几秒，默认 `300`。

### 3. 验证代码能导入

```powershell
python -m py_compile main.py graph.py tool.py daily_email.py outlook_mcp.py schedular.py
```

### 4. 初始化 Outlook 登录

Outlook 邮件发送依赖 Microsoft Graph 设备码登录。登录成功后，本地会生成 `.outlook_token_cache.json`，之后不需要每次重新登录，除非 token 失效或被删除。

先启动 Outlook MCP server：

```powershell
python outlook_mcp.py
```

然后在 MCP 客户端中按顺序调用：

1. `start_outlook_login`
2. 打开返回的 `verification_uri`
3. 输入返回的 `user_code`
4. 调用 `finish_outlook_login`

登录完成后可以检查白名单配置：

```powershell
python -c "from outlook_mcp import get_mail_whitelist; print(get_mail_whitelist())"
```

### 5. 交互式运行 Research Agent

```powershell
python main.py
```

输入研究型问题时，Agent 会搜索、读取网页并生成 Markdown 笔记。输入简单问题时，Agent 应该直接回答，不保存笔记。

退出：

```text
exit
```

### 6. 手动发送一次每日新闻邮件

```powershell
python daily_email.py
```

这个命令会：

1. 用 Tavily 搜索当天新闻，固定参数为 `days=1`、`topic=news`
2. 批量读取搜索到的网页
3. 生成当天 Research Markdown
4. 保存到 `outputs/`
5. 发送纯文本 Outlook 邮件到 `DAILY_EMAIL_TO`
6. 写入运行日志

如果 Tavily、LLM、爬虫或 Outlook 临时失败，`daily_email.py` 会按 `.env` 中的重试配置自动重试。每一次尝试都会写入 `outputs/runs.jsonl`。

### 7. 启动常驻定时任务

```powershell
python schedular.py
```

默认配置是每天 `08:00` 运行一次，每 `30s` 检查一次时间。可以通过 `.env` 修改：

```env
DAILY_EMAIL_HOUR=8
DAILY_EMAIL_MINUTE=0
DAILY_EMAIL_CHECK_INTERVAL_SECONDS=30
```

如果 `daily_email()` 运行失败，`schedular.py` 会打印错误并继续等待下一轮，不会因为一次失败直接退出。

## 输出文件

运行产物默认写入 `outputs/`：

- `outputs/*.md`：生成的 Research Markdown 笔记。
- `outputs/runs.jsonl`：每次运行的状态记录。
- `outputs/index.jsonl`：笔记索引，包含 `note_id`、路径、摘要、双链和 sources。

这些运行产物默认不提交到 Git。

## 常见问题

### 缺少 `DEEPSEEK_API_KEY`

检查 `.env` 是否存在，并确认写了：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 缺少 `TAVILY_API_KEY`

检查 `.env` 是否存在，并确认写了：

```env
TAVILY_API_KEY=your_tavily_api_key
```

### Outlook 尚未登录

先完成 Outlook MCP 的设备码登录。登录缓存文件是 `.outlook_token_cache.json`，它只保存在本地，不应该提交。

### 邮箱不在白名单内

确保 `DAILY_EMAIL_TO` 中的邮箱也出现在 `MAIL_WHITELIST` 中。
