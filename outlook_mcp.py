from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from msal import PublicClientApplication, SerializableTokenCache


load_dotenv()

WORK_DIR = Path(__file__).parent
TOKEN_CACHE_PATH = WORK_DIR / ".outlook_token_cache.json"
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.Send", "Mail.Read"]

mcp = FastMCP("outlook-mail")
_device_flow: dict[str, Any] | None = None


def _mail_whitelist() -> set[str]:
    raw = os.getenv("MAIL_WHITELIST", "")
    configured = set(_parse_emails(raw))
    if not configured:
        raise RuntimeError("缺少环境变量 MAIL_WHITELIST")
    return configured


def _parse_emails(value: str) -> list[str]:
    emails = [
        item.strip().lower()
        for item in value.replace(";", ",").split(",")
        if item.strip()
    ]
    return sorted(set(emails))


def _ensure_whitelisted(emails: list[str]) -> None:
    whitelist = _mail_whitelist()
    blocked = [email for email in emails if email not in whitelist]
    if blocked:
        raise PermissionError(f"邮箱不在白名单内：{', '.join(blocked)}")


def _load_cache() -> SerializableTokenCache:
    cache = SerializableTokenCache()
    if TOKEN_CACHE_PATH.exists():
        cache.deserialize(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
    return cache


def _save_cache(cache: SerializableTokenCache) -> None:
    if cache.has_state_changed:
        TOKEN_CACHE_PATH.write_text(cache.serialize(), encoding="utf-8")


def _client_id() -> str:
    client_id = os.getenv("MS_CLIENT_ID", "").strip()
    if not client_id:
        raise RuntimeError("缺少环境变量 MS_CLIENT_ID")
    return client_id


def _tenant_id() -> str:
    return os.getenv("MS_TENANT_ID", "consumers").strip() or "consumers"


def _make_app(cache: SerializableTokenCache) -> PublicClientApplication:
    return PublicClientApplication(
        client_id=_client_id(),
        authority=f"https://login.microsoftonline.com/{_tenant_id()}",
        token_cache=cache,
    )


def _get_access_token() -> str:
    cache = _load_cache()
    app = _make_app(cache)
    accounts = app.get_accounts()

    result = None
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])

    _save_cache(cache)

    if not result or "access_token" not in result:
        raise RuntimeError("Outlook 尚未登录。请先调用 start_outlook_login，再调用 finish_outlook_login。")

    return result["access_token"]


def _graph_request(method: str, path: str, **kwargs: Any) -> Any:
    token = _get_access_token()
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    headers["Content-Type"] = "application/json"

    response = requests.request(
        method,
        f"{GRAPH_BASE_URL}{path}",
        headers=headers,
        timeout=30,
        **kwargs,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Microsoft Graph 请求失败：{response.status_code} {response.text}")

    if not response.content:
        return {}

    return response.json()


@mcp.tool()
def start_outlook_login() -> dict[str, Any]:
    """开始 Outlook 设备码登录，返回需要在浏览器中输入的 code。"""

    global _device_flow

    cache = _load_cache()
    app = _make_app(cache)
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError(f"无法创建 Outlook 登录流程：{json.dumps(flow, ensure_ascii=False)}")

    _device_flow = flow
    return {
        "verification_uri": flow.get("verification_uri"),
        "user_code": flow.get("user_code"),
        "expires_in": flow.get("expires_in"),
        "message": flow.get("message"),
    }


@mcp.tool()
def finish_outlook_login() -> dict[str, Any]:
    """完成 Outlook 设备码登录。先调用 start_outlook_login，并在浏览器中输入 code。"""

    global _device_flow

    if not _device_flow:
        raise RuntimeError("没有正在进行的登录流程。请先调用 start_outlook_login。")

    cache = _load_cache()
    app = _make_app(cache)
    result = app.acquire_token_by_device_flow(_device_flow) 
    _device_flow = None
    _save_cache(cache)

    if "access_token" not in result:
        raise RuntimeError(f"Outlook 登录失败：{json.dumps(result, ensure_ascii=False)}")

    return {"ok": True, "account": result.get("id_token_claims", {}).get("preferred_username")}


@mcp.tool()
def send_outlook_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """发送 Outlook 邮件。to 必须是白名单邮箱，多个地址用英文逗号分隔。"""

    recipients = _parse_emails(to)
    if not recipients:
        raise ValueError("to 不能为空")
    _ensure_whitelisted(recipients)

    subject = subject.strip()
    body = body.strip()
    if not subject:
        raise ValueError("subject 不能为空")
    if not body:
        raise ValueError("body 不能为空")

    payload = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "Text",
                "content": body,
            },
            "toRecipients": [
                {"emailAddress": {"address": email}}
                for email in recipients
            ],
        },
        "saveToSentItems": True,
    }

    _graph_request("POST", "/me/sendMail", json=payload)
    return {"ok": True, "to": recipients, "subject": subject}


@mcp.tool()
def read_whitelisted_outlook_emails(limit: int = 10, unread_only: bool = True) -> list[dict[str, Any]]:
    """读取 Outlook 收件箱，只返回白名单发件人的邮件。"""

    limit = max(1, min(int(limit), 50))
    params: dict[str, Any] = {
        "$top": limit,
        "$select": "id,subject,from,receivedDateTime,bodyPreview,isRead",
        "$orderby": "receivedDateTime desc",
    }
    if unread_only:
        params["$filter"] = "isRead eq false"

    data = _graph_request(
        "GET",
        "/me/mailFolders/inbox/messages",
        params=params,
        headers={"Prefer": 'outlook.body-content-type="text"'},
    )

    whitelist = _mail_whitelist()
    messages: list[dict[str, Any]] = []
    for item in data.get("value", []):
        sender = (
            item.get("from", {})
            .get("emailAddress", {})
            .get("address", "")
            .strip()
            .lower()
        )
        if sender not in whitelist:
            continue

        messages.append(
            {
                "id": item.get("id"),
                "from": sender,
                "subject": item.get("subject"),
                "receivedDateTime": item.get("receivedDateTime"),
                "bodyPreview": item.get("bodyPreview"),
                "isRead": item.get("isRead"),
            }
        )

    return messages


@mcp.tool()
def get_mail_whitelist() -> list[str]:
    """查看当前允许发送和接收的邮箱白名单。"""

    return sorted(_mail_whitelist())


if __name__ == "__main__":
    mcp.run()
