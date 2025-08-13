# voice_agent_manager.py
import asyncio, time, contextlib
from typing import Optional, Callable, Dict, Any
from urllib.parse import urlparse
from uuid import uuid4
import httpx
import psutil


# ===== Registry =====
class ServiceEntry:
    __slots__ = (
        "id", "name", "config", "health_url",
        "state", "error", "started_at", "ready_at",
        "server_task", "monitor_task", "stop_fn",
    )
    def __init__(self, name: str, config: Optional[Dict[str, Any]], health_url: str, stop_fn: Optional[Callable]=None):
        self.id = f"{name}-{uuid4().hex[:8]}"
        self.name = name
        self.config = config
        self.health_url = health_url
        self.state = "starting"        # starting | ready | timeout | failed | stopped | cancelled
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.ready_at: Optional[float] = None
        self.server_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.stop_fn = stop_fn

services: dict[str, ServiceEntry] = {}

# ===== Public APIs bạn sẽ gọi từ FastAPI =====
async def start_service(
    name: str,
    config: Optional[Dict[str, Any]],
    run_fn_blocking,              # ví dụ run_stt_app_func (sync & blocking)
    health_url: str = None,
    stop_fn: Optional[Callable] = None,  # ví dụ stop_stt_app (tuỳ chọn)
) -> str:
    entry = ServiceEntry(name, config, health_url, stop_fn)
    services[entry.id] = entry

    # chạy hàm blocking ở thread → không khoá event loop
    entry.server_task = asyncio.create_task(asyncio.to_thread(run_fn_blocking, config))
    return entry.id

async def service_status(service_id: str, health_timeout: float = 30.0) -> dict:
    e = services.get(service_id)
    await _monitor_health(service_id, health_timeout)
    if not e:
        return {"error": "service not found"}
    # cập nhật state nếu server_task đã kết thúc
    try:
        if e.server_task and e.server_task.done() and e.state in ("starting",):
            exc = e.server_task.exception()
            e.state = "failed" if exc else "stopped"
            e.error = str(exc) if exc else None
        return {
            "id": e.id, "name": e.name, "model": e.model,
            "state": e.state, "error": e.error,
            "started_at": e.started_at, "ready_at": e.ready_at,
            "health_url": e.health_url,
        }
    except Exception as e:
        print(f"[manager] service_status error: {e}")
        return {"error": f"failed to get service status: {str(e)}"}

async def list_services() -> dict:
    return {sid: await service_status(sid) for sid in list(services.keys())}

async def cancel_service(service_id: str, *, kill_by_port: Optional[int] = None) -> dict:
    e = services.get(service_id)
    if not e:
        return {"error": "service not found"}

    # 1) thử stop_fn nếu có (khuyên dùng – xem mục 2 bên dưới)
    if e.stop_fn:
        await asyncio.to_thread(e.stop_fn)
    # 2) nếu không có stop_fn mà bạn biết port, kill theo port (best-effort)
    elif kill_by_port is not None:
        _best_effort_kill_by_port(kill_by_port)

    # dừng monitor
    if e.monitor_task and not e.monitor_task.done():
        e.monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await e.monitor_task

    # đợi thread kết thúc một chút (nếu run_fn thoát sau khi kill)
    if e.server_task and not e.server_task.done():
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(e.server_task, timeout=5.0)

    # cập nhật trạng thái cuối
    if e.server_task and e.server_task.done():
        exc = e.server_task.exception()
        e.state = "failed" if exc else "stopped"
        e.error = str(exc) if exc else None
    else:
        e.state = "cancelled"
    return await service_status(service_id)

# ===== Helpers =====
async def _monitor_health(service_id: str, timeout_s: float) -> None:
    e = services.get(service_id)
    if not e or not e.health_url:
        print(f"[manager] health-check skipped for {service_id}: no health_url")
        return

    deadline = time.monotonic() + max(1, timeout_s or 30)
    last_err = None
    backoff = 0.5

    # tắt verify nếu là https
    verify_flag = False if urlparse(e.health_url).scheme == "https" else True

    # timeout chi tiết để tránh treo connect/read
    client_timeout = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=client_timeout,
        # nếu bạn dùng HTTPS tự ký, tắt verify (chỉ khi môi trường tin cậy):
        verify=verify_flag,
    ) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(e.health_url)
                if r.status_code == 200:
                    # tùy bạn: có thể kiểm tra payload: r.json().get("status") == "ok"
                    e.state = "ready"
                    e.ready_at = time.time()
                    e.error = None
                    return
                else:
                    last_err = f"HTTP {r.status_code}"
            except Exception as ex:
                print(f"[manager] health-check error: {ex}")
                last_err = str(ex)

            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.7, 5.0)

    e.state = "timeout"
    e.error = last_err or "health-check timed out"

def _best_effort_kill_by_port(port: int):
    """Không cần sửa code cũ: kill process bằng port (Linux/macOS)."""
    try:
        for p in psutil.process_iter(attrs=["pid", "name"]):
            with contextlib.suppress(Exception):
                for c in p.net_connections(kind="inet"):
                    if c.laddr and c.laddr.port == port:
                        p.terminate()
                        try:
                            p.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            p.kill()
    except Exception as e:
        print(f"[manager] kill-by-port failed: {e}")
