"""Local P10 forecast website. Run with the Python environment that has PyTorch."""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import urlsplit

if __name__ == "__main__":
    print("Loading prediction dependencies (the first start may take a minute)...", flush=True)

try:
    import numpy as np
    import pandas as pd
    import torch

    import forecast_core as core
except (ImportError, OSError) as exc:
    print(f"Could not load prediction dependencies: {exc}", file=sys.stderr)
    print(f'Python: {sys.executable}', file=sys.stderr)
    print("Run the dependency installer in this project folder, then try again.", file=sys.stderr)
    raise SystemExit(1) from None


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MODELS = ROOT / "models"
SAMPLES = ROOT / "samples"
RESOLUTIONS = ("5min", "1h", "2h", "1d", "3d")
LABELS = {"5min": "5 分钟", "1h": "1 小时", "2h": "2 小时", "1d": "1 天", "3d": "3 天"}
MAX_BODY = 24 * 1024 * 1024
MAX_ROWS = 300_000

torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


@lru_cache(maxsize=5)
def load_model(resolution: str):
    if resolution not in RESOLUTIONS:
        raise ValueError("不支持的预测尺度")
    path = MODELS / f"p10_{resolution}_verified.pt"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {LABELS[resolution]} 模型")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["resolution"] != resolution or tuple(config["features"]) != core.FEATURES:
        raise ValueError(f"{LABELS[resolution]} 模型配置不匹配")
    model = core.build_model(int(config["lookback"]), int(config["steps"]), int(config["d_model"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def clean_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"datetime", *core.FEATURES}.difference(frame.columns)
    if missing:
        raise ValueError("CSV 缺少列：" + ", ".join(sorted(missing)))
    if len(frame) > MAX_ROWS:
        raise ValueError(f"最多允许 {MAX_ROWS:,} 行")
    data = frame.loc[:, ["datetime", *core.FEATURES]].copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    for col in core.FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values("datetime").drop_duplicates("datetime", keep="last")
    if data.empty:
        raise ValueError("CSV 中没有有效记录")
    if (data.loc[:, core.LOG_FEATURES] < 0).any().any():
        raise ValueError("E2、P1、P10 不能为负值")
    return data.reset_index(drop=True)


def aggregate_5min(frame: pd.DataFrame, resolution: str) -> pd.DataFrame:
    if resolution == "5min":
        return frame
    dates = pd.DatetimeIndex(frame["datetime"])
    if ((dates.minute % 5) != 0).any() or (dates.second != 0).any():
        raise ValueError("自动生成多个尺度时，输入时间戳必须落在整 5 分钟")
    rule = core.FREQUENCIES[resolution]
    origin = pd.Timestamp("2022-03-29") if resolution == "3d" else "start_day"
    indexed = frame.set_index("datetime")
    grouped = indexed.resample(rule, origin=origin)
    counts = grouped.size()
    values = grouped.max()
    expected = int(pd.Timedelta(rule) / pd.Timedelta("5min"))
    complete = values.loc[counts == expected]
    if complete.empty:
        raise ValueError(f"没有完整的 {LABELS[resolution]} 时间区间")
    return complete.reset_index()


def predict_frame(frame: pd.DataFrame, resolution: str, *, historical: bool, timezone: str):
    checkpoint, model = load_model(resolution)
    prepared = core.prepare_dataframe(clean_raw_frame(frame))
    config = checkpoint["config"]
    lookback = int(config["lookback"])
    if len(prepared) < lookback:
        raise ValueError(f"至少需要 {lookback} 条连续的 {LABELS[resolution]}记录；当前只有 {len(prepared)} 条")
    recent = prepared.iloc[-lookback:]
    times = pd.DatetimeIndex(recent["datetime"])
    interval = pd.Timedelta(core.FREQUENCIES[resolution])
    if not (times[1:] - times[:-1] == interval).all():
        raise ValueError(f"最后 {lookback} 条记录有时间缺口，无法按 {LABELS[resolution]}预测")
    latest = times[-1]
    core.validate_freshness(latest, resolution, allow_stale=historical, timezone=timezone)
    features = core.transform_features(recent, checkpoint["feature_scaler"])
    with torch.inference_mode():
        scaled = model(torch.from_numpy(features).unsqueeze(0)).numpy()[0]
    flux = core.inverse_target(scaled, checkpoint["target_scaler"])
    if not np.isfinite(flux).all() or (flux < 0).any():
        raise ValueError("模型返回了无效预测值")
    baseline = float(10 ** recent["p10"].iloc[-1])
    predictions = [{
        "datetime": (latest + (index + 1) * interval).isoformat(),
        "step": index + 1,
        "flux": float(value),
        "alarm": core.alarm_level(float(value)),
    } for index, value in enumerate(flux)]
    series = [{"datetime": timestamp.isoformat(), "flux": float(10 ** value)}
              for timestamp, value in zip(times[-36:], recent["p10"].to_numpy()[-36:])]
    metrics = checkpoint.get("metrics", {})
    return {
        "resolution": resolution,
        "label": LABELS[resolution],
        "last_observation": latest.isoformat(),
        "timezone": timezone,
        "input_count": len(prepared),
        "lookback": lookback,
        "baseline_flux": baseline,
        "baseline_alarm": core.alarm_level(baseline),
        "predictions": predictions,
        "recent_series": series,
        "test_r2_log": float(metrics["r2_log"]) if "r2_log" in metrics else None,
        "test_rmse_log": float(metrics["rmse_log"]) if "rmse_log" in metrics else None,
        "historical": historical,
    }


def run_request(payload: dict) -> dict:
    mode = payload.get("mode", "direct")
    historical = payload.get("historical") is True
    timezone = payload.get("timezone", "UTC")
    if timezone not in ("UTC", "Asia/Shanghai"):
        raise ValueError("请选择 UTC 或北京时间")
    csv_text = payload.get("csv")
    if not isinstance(csv_text, str) or not csv_text.strip():
        raise ValueError("请上传或粘贴 CSV 数据")
    if len(csv_text.encode("utf-8")) > MAX_BODY:
        raise ValueError("CSV 文件过大")
    frame = clean_raw_frame(pd.read_csv(io.StringIO(csv_text)))
    if mode == "direct":
        resolution = payload.get("resolution")
        if resolution not in RESOLUTIONS:
            raise ValueError("请选择有效的时间尺度")
        return {"mode": mode, "results": {resolution: predict_frame(frame, resolution, historical=historical, timezone=timezone)}, "errors": {}}
    if mode != "auto":
        raise ValueError("不支持的输入模式")
    results, errors = {}, {}
    for resolution in RESOLUTIONS:
        try:
            source = aggregate_5min(frame, resolution)
            results[resolution] = predict_frame(source, resolution, historical=historical, timezone=timezone)
        except (ValueError, FileNotFoundError) as exc:
            errors[resolution] = str(exc)
    return {"mode": mode, "results": results, "errors": errors}


def demo_request() -> dict:
    results, errors = {}, {}
    for resolution in RESOLUTIONS:
        try:
            frame = pd.read_csv(SAMPLES / f"{resolution}.csv")
            results[resolution] = predict_frame(frame, resolution, historical=True, timezone="UTC")
        except (ValueError, FileNotFoundError) as exc:
            errors[resolution] = str(exc)
    return {"mode": "demo", "results": results, "errors": errors}


class Handler(BaseHTTPRequestHandler):
    server_version = "P10Local/1.0"

    def send_bytes(self, status: int, content: bytes, mime: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status: int, value: dict) -> None:
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_bytes(status, content, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/status":
            models = [{"resolution": item, "label": LABELS[item], "lookback": core.DEFAULT_LOOKBACK[item],
                       "available": (MODELS / f"p10_{item}_verified.pt").is_file()}
                      for item in RESOLUTIONS]
            self.send_json(200, {"models": models, "experimental": True})
            return
        if path == "/api/demo":
            self.send_json(200, demo_request())
            return
        files = {
            "/": (STATIC / "index.html", "text/html; charset=utf-8"),
            "/index.html": (STATIC / "index.html", "text/html; charset=utf-8"),
            "/styles.css": (STATIC / "styles.css", "text/css; charset=utf-8"),
            "/app.js": (STATIC / "app.js", "text/javascript; charset=utf-8"),
        }
        if path not in files:
            self.send_json(404, {"error": "页面不存在"})
            return
        file, mime = files[path]
        self.send_bytes(200, file.read_bytes(), mime)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/predict":
            self.send_json(404, {"error": "接口不存在"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY:
                self.send_json(413, {"error": "请求内容为空或超过 24 MB"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求格式无效")
            self.send_json(200, run_request(payload))
        except (ValueError, KeyError, pd.errors.ParserError, FileNotFoundError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            traceback.print_exc()
            self.send_json(500, {"error": "推理时发生内部错误，请查看本地终端日志"})


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 P10 预测网站")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true", help="服务就绪后打开浏览器")
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"Could not start http://{args.host}:{args.port}: {exc}\n"
                    "If the site is already running, open its URL. Otherwise stop the process using this port.\n")
    url = f"http://{args.host}:{server.server_port}"
    print(f"P10 forecast site is ready: {url}", flush=True)
    print("Keep this process running. Press Ctrl+C to stop.", flush=True)
    if args.open_browser:
        browser_timer = Timer(0.5, webbrowser.open, args=(url,))
        browser_timer.daemon = True
        browser_timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
