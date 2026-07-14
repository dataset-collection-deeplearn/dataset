"""
读取并分析 Sensor.U8.XY.h5 目录下的全部 HDF5 数据集。

运行方式：
    python read_datasets.py
    python read_datasets.py --preview-points 5 --preview-times 5

输出：
    dataset_overview.md              中文总览报告
    dataset_summary.csv              每个 HDF5 内部数据集的形状和值域统计
    dataset_previews/*.csv           每个 HDF5 文件的前若干空间点/时间步样本
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DATASET_MEANING = {
    "x_all": "无量纲 x 坐标；x = X / L，L=60 m。",
    "y_all": "无量纲 y 坐标；y = Y / L，L=60 m。",
    "t_all": "无量纲时间；t = T / (L/U)，L=60 m，U=8 m/s。",
    "u_all": "无量纲流向速度 u；u = U_x / U_ref，U_ref=8 m/s。",
    "v_all": "无量纲横向速度 v；v = U_y / U_ref，U_ref=8 m/s。",
    "p_all": "无量纲压力 p；用于压力观测或完整参考场。",
    "u_LoS_all": "LiDAR 视线方向风速观测，已由 u、v 投影到光束方向。",
    "u_mag_all": "速度模值观测，sqrt(u^2+v^2)。",
    "u_dir_all": "速度方向角观测，通常由 atan(v/u) 得到。",
    "u_mag": "速度模值观测的重复/兼容字段，内容与 u_mag_all 一致。",
    "u_dir": "速度方向角观测的重复/兼容字段，内容与 u_dir_all 一致。",
}


def role_from_name(name: str) -> str:
    if name.startswith("Center_XY"):
        if "100-199s" in name:
            return "完整参考场/后续时间窗；可用于未来时间测试或迁移学习评估"
        return "完整参考场/验证网格；用于评估重构后的全场误差"
    if name.startswith("LIDAR_Beam_Up"):
        return "LiDAR 上束稀疏观测；主要用于 LoS 数据损失"
    if name.startswith("LIDAR_Beam_Down"):
        return "LiDAR 下束稀疏观测；主要用于 LoS 数据损失"
    if name.startswith("M_XY_Mid"):
        return "中线测风塔稀疏观测；代码中用于速度矢量 Uvec 数据损失"
    if name.startswith("M_XY_Down"):
        return "下侧测风塔稀疏观测；代码中用于速度分量 uv 数据损失"
    if name.startswith("M_XY_Up"):
        return "上侧测风塔稀疏观测；代码中用于压力 p 数据损失"
    return "未识别用途，请结合键名检查"


def as_array(value: Any) -> np.ndarray:
    return np.asarray(value)


def flat_numeric_stats(arr: np.ndarray) -> dict[str, Any]:
    flat = arr.reshape(-1)
    if flat.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return {
            "min": "",
            "max": "",
            "mean": "",
            "std": "",
            "first_values": "",
        }
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {
            "min": "nan",
            "max": "nan",
            "mean": "nan",
            "std": "nan",
            "first_values": "",
        }
    return {
        "min": f"{float(finite.min()):.8g}",
        "max": f"{float(finite.max()):.8g}",
        "mean": f"{float(finite.mean()):.8g}",
        "std": f"{float(finite.std()):.8g}",
        "first_values": ", ".join(f"{float(v):.6g}" for v in flat[:6]),
    }


def infer_grid_info(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if "x_all" not in arrays or "y_all" not in arrays:
        return info

    x = arrays["x_all"].reshape(-1)
    y = arrays["y_all"].reshape(-1)
    ux = np.unique(np.round(x, 8))
    uy = np.unique(np.round(y, 8))
    info["n_space"] = x.size
    info["x_range"] = (float(x.min()), float(x.max()))
    info["y_range"] = (float(y.min()), float(y.max()))
    info["unique_x"] = ux.size
    info["unique_y"] = uy.size
    if ux.size > 1:
        dx = np.diff(np.sort(ux))
        info["dx_minmax"] = (float(dx.min()), float(dx.max()))
    if uy.size > 1:
        dy = np.diff(np.sort(uy))
        info["dy_minmax"] = (float(dy.min()), float(dy.max()))
    if ux.size * uy.size == x.size:
        info["structured_grid"] = f"{ux.size} x {uy.size}"
    else:
        info["structured_grid"] = "不是完整张量网格；是稀疏测点或斜向光束点"

    if "t_all" in arrays:
        t = arrays["t_all"].reshape(-1)
        info["n_time"] = t.size
        info["t_range"] = (float(t.min()), float(t.max()))
        if t.size > 1:
            dt = np.diff(t)
            info["dt_minmax"] = (float(dt.min()), float(dt.max()))
    return info


def collect_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        for key in sorted(h5.keys()):
            arr = as_array(h5[key])
            arrays[key] = arr
            stats = flat_numeric_stats(arr)
            rows.append(
                {
                    "file": path.name,
                    "file_role": role_from_name(path.name),
                    "dataset": key,
                    "meaning": DATASET_MEANING.get(key, "未在脚本中登记含义，请结合数据形状判断"),
                    "shape": "x".join(str(v) for v in arr.shape),
                    "dtype": str(arr.dtype),
                    **stats,
                }
            )
    return rows, infer_grid_info(arrays), arrays


def write_preview(path: Path, arrays: dict[str, np.ndarray], out_dir: Path, preview_points: int, preview_times: int) -> None:
    if not {"x_all", "y_all", "t_all"}.issubset(arrays):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    x = arrays["x_all"].reshape(-1)
    y = arrays["y_all"].reshape(-1)
    t = arrays["t_all"].reshape(-1)
    variables = [
        key
        for key, arr in arrays.items()
        if key not in {"x_all", "y_all", "t_all"} and arr.ndim == 2 and arr.shape[0] == x.size and arr.shape[1] == t.size
    ]
    out_path = out_dir / f"{path.stem}_preview.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["space_index", "time_index", "x", "y", "t", *variables])
        for i in range(min(preview_points, x.size)):
            for j in range(min(preview_times, t.size)):
                writer.writerow([i, j, x[i], y[i], t[j], *[arrays[key][i, j] for key in variables]])


def write_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    headers = ["file", "file_role", "dataset", "meaning", "shape", "dtype", "min", "max", "mean", "std", "first_values"]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def fmt_range(value: Any) -> str:
    if isinstance(value, tuple):
        return f"[{value[0]:.8g}, {value[1]:.8g}]"
    return str(value)


def write_overview(files: list[Path], all_rows: list[dict[str, Any]], file_infos: dict[str, dict[str, Any]], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Sensor.U8.XY.h5 数据集读取总览")
    lines.append("")
    lines.append("本报告由 `read_datasets.py` 自动生成。所有坐标和物理量均按项目代码中的无量纲形式保存。")
    lines.append("")
    lines.append("## 文件级总览")
    lines.append("")
    lines.append("| 文件 | 作用 | 空间点数 | 时间步 | x范围 | y范围 | t范围 | 网格/测点结构 |")
    lines.append("|---|---|---:|---:|---|---|---|---|")
    for path in files:
        info = file_infos[path.name]
        lines.append(
            "| "
            + " | ".join(
                [
                    path.name,
                    role_from_name(path.name),
                    str(info.get("n_space", "")),
                    str(info.get("n_time", "")),
                    fmt_range(info.get("x_range", "")),
                    fmt_range(info.get("y_range", "")),
                    fmt_range(info.get("t_range", "")),
                    str(info.get("structured_grid", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 每个 HDF5 内部数据集")
    lines.append("")
    for path in files:
        lines.append(f"### {path.name}")
        lines.append("")
        lines.append(f"用途：{role_from_name(path.name)}")
        info = file_infos[path.name]
        if info:
            lines.append(
                f"空间点：{info.get('n_space', '')}；时间步：{info.get('n_time', '')}；"
                f"x范围：{fmt_range(info.get('x_range', ''))}；y范围：{fmt_range(info.get('y_range', ''))}；"
                f"t范围：{fmt_range(info.get('t_range', ''))}；结构：{info.get('structured_grid', '')}。"
            )
        lines.append("")
        lines.append("| 键名 | 含义 | shape | dtype | min | max | mean | std | 前6个值 |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---|")
        for row in [r for r in all_rows if r["file"] == path.name]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["dataset"],
                        row["meaning"],
                        row["shape"],
                        row["dtype"],
                        row["min"],
                        row["max"],
                        row["mean"],
                        row["std"],
                        row["first_values"],
                    ]
                )
                + " |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def print_brief(files: list[Path], file_infos: dict[str, dict[str, Any]]) -> None:
    print("\nSensor.U8.XY.h5 数据读取完成\n")
    for path in files:
        info = file_infos[path.name]
        print(f"- {path.name}")
        print(f"  作用: {role_from_name(path.name)}")
        print(
            "  "
            f"空间点={info.get('n_space', '')}, 时间步={info.get('n_time', '')}, "
            f"x={fmt_range(info.get('x_range', ''))}, y={fmt_range(info.get('y_range', ''))}, "
            f"t={fmt_range(info.get('t_range', ''))}, 结构={info.get('structured_grid', '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 Sensor.U8.XY.h5 目录下所有 HDF5 数据集并输出中文摘要。")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent, help="HDF5 数据目录")
    parser.add_argument("--preview-points", type=int, default=5, help="每个文件预览的空间点数量")
    parser.add_argument("--preview-times", type=int, default=5, help="每个文件预览的时间步数量")
    args = parser.parse_args()

    data_dir = args.data_dir
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 下找到 .h5 文件")

    all_rows: list[dict[str, Any]] = []
    file_infos: dict[str, dict[str, Any]] = {}
    preview_dir = data_dir / "dataset_previews"

    for path in files:
        rows, info, arrays = collect_file(path)
        all_rows.extend(rows)
        file_infos[path.name] = info
        write_preview(path, arrays, preview_dir, args.preview_points, args.preview_times)

    summary_csv = data_dir / "dataset_summary.csv"
    overview_md = data_dir / "dataset_overview.md"
    write_summary_csv(all_rows, summary_csv)
    write_overview(files, all_rows, file_infos, overview_md)
    print_brief(files, file_infos)
    print(f"\n已输出: {summary_csv}")
    print(f"已输出: {overview_md}")
    print(f"已输出预览 CSV 目录: {preview_dir}")


if __name__ == "__main__":
    main()
