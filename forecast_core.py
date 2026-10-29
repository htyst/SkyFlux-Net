"""Train and run a reproducible P10 time-series forecast.

Examples:
    python p10_forecast.py train --resolution 1h
    python p10_forecast.py predict --resolution 1h --data path/to/current.csv
    python p10_forecast.py predict --resolution 1h --allow-stale
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURES = ("bulkspeed", "kp", "e2", "p1", "p10")
LOG_FEATURES = ("e2", "p1", "p10")
FREQUENCIES = {"5min": "5min", "1h": "1h", "2h": "2h", "1d": "1D", "3d": "3D"}
DEFAULT_LOOKBACK = {"5min": 48, "1h": 24, "2h": 12, "1d": 12, "3d": 6}
MIN_POSITIVE = 1e-6


def default_data_path(resolution: str) -> Path:
    return ROOT / "data" / f"data_all_{resolution}_max.csv"


def default_model_path(resolution: str) -> Path:
    return ROOT / "result" / "p10" / f"p10_{resolution}_verified.pt"


def read_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    return prepare_dataframe(pd.read_csv(path))


def prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    needed = {"datetime", *FEATURES}
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError(f"数据缺少列：{', '.join(sorted(missing))}")
    data = data.loc[:, ["datetime", *FEATURES]].copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values("datetime").drop_duplicates("datetime", keep="last")
    if data.empty:
        raise ValueError("数据清理后为空")
    if (data.loc[:, LOG_FEATURES] < 0).any().any():
        raise ValueError("E2、P1、P10 不能有负值")
    for col in LOG_FEATURES:
        data[col] = np.log10(data[col].clip(lower=MIN_POSITIVE))
    return data.reset_index(drop=True)


def split_chronologically(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    train_end, valid_end = int(n * 0.70), int(n * 0.85)
    return data.iloc[:train_end].copy(), data.iloc[train_end:valid_end].copy(), data.iloc[valid_end:].copy()


def fit_scalers(train: pd.DataFrame) -> tuple[StandardScaler, StandardScaler]:
    feature_scaler = StandardScaler().fit(train.loc[:, FEATURES])
    target_scaler = StandardScaler().fit(train.loc[:, ["p10"]])
    return feature_scaler, target_scaler


def scaler_info(scaler: StandardScaler) -> dict[str, list[float]]:
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def transform_features(data: pd.DataFrame, info: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(info["mean"], dtype=np.float64)
    scale = np.asarray(info["scale"], dtype=np.float64)
    if mean.shape != (len(FEATURES),) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("模型中的特征标准化参数无效")
    return ((data.loc[:, FEATURES].to_numpy(dtype=np.float64) - mean) / scale).astype(np.float32)


def transform_target(data: pd.DataFrame, info: dict[str, list[float]]) -> np.ndarray:
    mean, scale = float(info["mean"][0]), float(info["scale"][0])
    if scale <= 0:
        raise ValueError("模型中的目标标准化参数无效")
    return ((data["p10"].to_numpy(dtype=np.float64) - mean) / scale).astype(np.float32)


def make_windows(
    data: pd.DataFrame,
    feature_info: dict[str, list[float]],
    target_info: dict[str, list[float]],
    lookback: int,
    steps: int,
    resolution: str,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    interval = pd.Timedelta(FREQUENCIES[resolution])
    values = transform_features(data, feature_info)
    targets = transform_target(data, target_info)
    timestamps = pd.DatetimeIndex(data["datetime"])
    x, y, prediction_times = [], [], []
    for start in range(len(data) - lookback - steps + 1):
        end = start + lookback + steps
        times = timestamps[start:end]
        if not (times[1:] - times[:-1] == interval).all():
            continue
        x.append(values[start:start + lookback])
        y.append(targets[start + lookback:end])
        prediction_times.append(times[lookback])
    return (
        np.asarray(x, dtype=np.float32).reshape(-1, lookback, len(FEATURES)),
        np.asarray(y, dtype=np.float32).reshape(-1, steps),
        pd.DatetimeIndex(prediction_times),
    )


def build_model(lookback: int, steps: int, d_model: int = 64):
    import torch
    from torch import nn

    class P10Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(len(FEATURES), d_model)
            position = torch.arange(lookback, dtype=torch.float32).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, lookback, d_model)
            pe[0, :, 0::2] = torch.sin(position * frequencies)
            pe[0, :, 1::2] = torch.cos(position * frequencies)
            self.register_buffer("position", pe)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, steps))

        def forward(self, x):
            encoded = self.encoder(self.projection(x) + self.position)
            return self.head(encoded[:, -1, :])

    return P10Transformer()


def alarm_level(flux: float) -> str:
    if flux >= 1000:
        return "红色"
    if flux >= 100:
        return "橙色"
    if flux >= 10:
        return "黄色"
    return "无告警"


def inverse_target(scaled: np.ndarray, info: dict[str, list[float]]) -> np.ndarray:
    log_flux = scaled * float(info["scale"][0]) + float(info["mean"][0])
    return np.power(10.0, log_flux)


def validate_freshness(
    latest: pd.Timestamp,
    resolution: str,
    max_age_intervals: int = 2,
    allow_stale: bool = False,
    now: pd.Timestamp | None = None,
    timezone: str = "UTC",
) -> None:
    if max_age_intervals < 1:
        raise ValueError("max_age_intervals 至少为 1")
    interval = pd.Timedelta(FREQUENCIES[resolution])
    now = pd.Timestamp.now(tz=timezone).tz_localize(None) if now is None else now
    age = now - latest
    if age > interval * max_age_intervals and not allow_stale:
        raise ValueError(f"数据已过期：最后观测为 {latest}，距今 {age}。请提供更新的 --data 文件；历史演示可加 --allow-stale")
    if age < -interval:
        raise ValueError(f"数据时间 {latest} 晚于当前时间，请检查时间戳或时区")


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    lookback = args.lookback or DEFAULT_LOOKBACK[args.resolution]
    if lookback < 2 or args.steps < 1:
        raise ValueError("lookback 至少为 2，steps 至少为 1")
    source = Path(args.data) if args.data else default_data_path(args.resolution)
    data = read_data(source)
    train_df, valid_df, test_df = split_chronologically(data)
    feature_scaler, target_scaler = fit_scalers(train_df)
    feature_info, target_info = scaler_info(feature_scaler), scaler_info(target_scaler)
    datasets = [make_windows(part, feature_info, target_info, lookback, args.steps, args.resolution)
                for part in (train_df, valid_df, test_df)]
    if any(len(x) == 0 for x, _, _ in datasets):
        raise ValueError("训练、验证或测试集没有连续时间窗口；请检查时间间隔和数据量")

    def loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                          batch_size=args.batch_size, shuffle=shuffle)

    train_loader = loader(*datasets[0][:2], shuffle=True)
    valid_loader = loader(*datasets[1][:2], shuffle=False)
    test_loader = loader(*datasets[2][:2], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; windows train/valid/test="
          f"{len(datasets[0][0])}/{len(datasets[1][0])}/{len(datasets[2][0])}")
    model = build_model(lookback, args.steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    best_loss, best_state, stale_epochs = float("inf"), None, 0

    for epoch in range(args.epochs):
        model.train()
        total_train = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * len(x)
        model.eval()
        total_valid = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                total_valid += criterion(model(x), y).item() * len(x)
        train_loss = total_train / len(train_loader.dataset)
        valid_loss = total_valid / len(valid_loader.dataset)
        print(f"Epoch {epoch + 1}: train={train_loss:.5f}, valid={valid_loss:.5f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("训练未产生有效模型")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            predictions.append(model(x.to(device)).cpu().numpy())
    pred_scaled = np.concatenate(predictions)
    true_scaled = datasets[2][1]
    pred_log = pred_scaled * target_info["scale"][0] + target_info["mean"][0]
    true_log = true_scaled * target_info["scale"][0] + target_info["mean"][0]
    true_flux, pred_flux = np.power(10, true_log), np.power(10, pred_log)
    true_alarm, pred_alarm = true_flux >= 10, pred_flux >= 10
    true_positives = int(np.logical_and(true_alarm, pred_alarm).sum())
    metrics = {
        "test_samples": int(len(pred_scaled)),
        "r2_log": float(r2_score(true_log.flatten(), pred_log.flatten())),
        "rmse_log": float(np.sqrt(mean_squared_error(true_log.flatten(), pred_log.flatten()))),
        "rmse_flux": float(np.sqrt(mean_squared_error(true_flux.flatten(), pred_flux.flatten()))),
        "alarm_true_count": int(true_alarm.sum()),
        "alarm_pred_count": int(pred_alarm.sum()),
        "alarm_precision": true_positives / int(pred_alarm.sum()) if pred_alarm.any() else None,
        "alarm_recall": true_positives / int(true_alarm.sum()) if true_alarm.any() else None,
    }
    model_path = Path(args.model) if args.model else default_model_path(args.resolution)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": {key: value.cpu() for key, value in best_state.items()},
        "config": {"resolution": args.resolution, "lookback": lookback, "steps": args.steps,
                   "d_model": 64, "features": list(FEATURES), "source": str(source.resolve()),
                   "train_end": str(train_df["datetime"].iloc[-1])},
        "feature_scaler": feature_info,
        "target_scaler": target_info,
        "metrics": metrics,
    }
    torch.save(checkpoint, model_path)
    rows = []
    interval = pd.Timedelta(FREQUENCIES[args.resolution])
    for i, first_time in enumerate(datasets[2][2]):
        for step in range(args.steps):
            rows.append({"datetime": first_time + step * interval, "step": step + 1,
                         "true_p10": true_flux[i, step], "pred_p10": pred_flux[i, step],
                         "pred_alert": alarm_level(float(pred_flux[i, step]))})
    result_path = model_path.with_suffix(".test_predictions.csv")
    pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"model": str(model_path), "test_predictions": str(result_path),
                      "metrics": metrics}, ensure_ascii=False, indent=2))


def predict(args: argparse.Namespace) -> None:
    import torch

    model_path = Path(args.model) if args.model else default_model_path(args.resolution)
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到经过修正后重新训练的模型：{model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "model_state_dict", "feature_scaler", "target_scaler"}.issubset(checkpoint):
        raise ValueError("该权重不是修正后训练产生的完整模型，请重新运行 train")
    config = checkpoint["config"]
    resolution = config["resolution"]
    if tuple(config["features"]) != FEATURES:
        raise ValueError("模型特征顺序与当前代码不一致")
    source = Path(args.data) if args.data else default_data_path(resolution)
    data = read_data(source)
    lookback = int(config["lookback"])
    interval = pd.Timedelta(FREQUENCIES[resolution])
    if len(data) < lookback:
        raise ValueError(f"至少需要 {lookback} 条数据")
    last = data.iloc[-lookback:]
    times = pd.DatetimeIndex(last["datetime"])
    if not (times[1:] - times[:-1] == interval).all():
        raise ValueError("最后一个输入窗口存在时间缺口")
    latest = times[-1]
    validate_freshness(latest, resolution, args.max_age_intervals, args.allow_stale,
                       timezone=args.data_timezone)
    features = transform_features(last, checkpoint["feature_scaler"])
    model = build_model(lookback, int(config["steps"]), int(config["d_model"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        scaled = model(torch.from_numpy(features).unsqueeze(0)).numpy()[0]
    flux = inverse_target(scaled, checkpoint["target_scaler"])
    rows = [{"datetime": latest + (step + 1) * interval, "step": step + 1,
             "pred_p10": float(value), "alert_level": alarm_level(float(value)),
             "last_observation": latest, "data_timezone": args.data_timezone,
             "data_source": str(source.resolve())}
            for step, value in enumerate(flux)]
    output = Path(args.output) if args.output else ROOT / "result" / "p10" / f"p10_{resolution}_forecast.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"结果已保存：{output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P10 数值时序预测（需使用修正后重新训练的模型）")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="按时间划分并训练")
    train_parser.add_argument("--resolution", choices=FREQUENCIES, default="1h")
    train_parser.add_argument("--data", help="输入 CSV，默认使用 data/data_all_<resolution>_max.csv")
    train_parser.add_argument("--model", help="输出模型路径")
    train_parser.add_argument("--lookback", type=int)
    train_parser.add_argument("--steps", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=42)
    predict_parser = subparsers.add_parser("predict", help="读取新数据并预测")
    predict_parser.add_argument("--resolution", choices=FREQUENCIES, default="1h")
    predict_parser.add_argument("--data", help="更新后的输入 CSV")
    predict_parser.add_argument("--model", help="train 命令产生的模型路径")
    predict_parser.add_argument("--output", help="预测结果 CSV 路径")
    predict_parser.add_argument("--max-age-intervals", type=int, default=2)
    predict_parser.add_argument("--data-timezone", default="UTC", help="CSV 时间戳的时区，默认 UTC")
    predict_parser.add_argument("--allow-stale", action="store_true", help="仅用于历史数据演示")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"错误：{exc}") from None
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("错误：当前 Python 环境没有 PyTorch，请安装依赖后重试") from None
        raise
