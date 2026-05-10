# -*- coding: utf-8 -*-
"""
plot_ablation_benchmark_runs.py

读取当前结构下的 benchmark_eval/public_benchmark_all_runs.csv，
按 instance_id 把不同变体画到同一张图里比较。

适配目录结构：
result/
  ablation_suite/
    baseline/
      benchmark_eval/
        public_benchmark_all_runs.csv
    ablation_no_zero_mask/
      benchmark_eval/
        public_benchmark_all_runs.csv
    ablation_no_fluid_state/
      benchmark_eval/
        public_benchmark_all_runs.csv
    ablation_no_reward_scale/
      benchmark_eval/
        public_benchmark_all_runs.csv
"""

import os
import math
import glob
import argparse

try:
    import pandas as pd
except ImportError:
    raise ImportError("需要安装 pandas：pip install pandas")

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 默认配置
# =========================================================
DEFAULT_RESULT_ROOT = os.path.join('result', 'ablation_suite')
DEFAULT_OUTPUT_DIR = os.path.join('result', 'ablation_suite', 'benchmark_eval_plots')

VARIANT_STYLE = {
    'baseline': {
        'label': 'FEGRL',
        'color': '#1f77b4'
    },
    'ablation_no_zero_mask': {
        'label': 'FEGRL-NM',
        'color': '#d62728'
    },
    'ablation_no_fluid_state': {
        'label': 'FEGRL-NS',
        'color': '#2ca02c'
    },
    'ablation_no_reward_scale': {
        'label': 'FEGRL-NR',
        'color': '#ff7f0e'
    }
}

DEFAULT_INSTANCES = ['P{}'.format(i) for i in range(1, 25)]

TITLE_FONTSIZE = 13
LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 9
TICK_FONTSIZE = 9

RAW_ALPHA = 0.18
RAW_LINEWIDTH = 0.8
SMOOTH_LINEWIDTH = 2.0
DEFAULT_SMOOTH_WINDOW = 5


# =========================================================
# 工具
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def moving_average(values, window=5):
    if values is None or len(values) == 0:
        return np.array([])

    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = np.mean(values[start:i + 1])

    return out


def scan_variant_csvs(result_root):
    """
    自动扫描所有变体的 benchmark_eval/public_benchmark_all_runs.csv
    返回：
      {
        'baseline': '.../baseline/benchmark_eval/public_benchmark_all_runs.csv',
        ...
      }
    """
    pattern = os.path.join(result_root, '*', 'benchmark_eval', 'public_benchmark_all_runs.csv')
    paths = glob.glob(pattern)

    variant_csv_map = {}
    for path in paths:
        variant_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        variant_csv_map[variant_name] = path

    return variant_csv_map


def load_all_benchmark_runs(result_root):
    """
    加载所有变体的 public_benchmark_all_runs.csv
    合并成一个 DataFrame
    """
    variant_csv_map = scan_variant_csvs(result_root)

    if len(variant_csv_map) == 0:
        raise FileNotFoundError(
            '未找到任何 benchmark_eval/public_benchmark_all_runs.csv，请检查目录: {}'.format(result_root)
        )

    all_dfs = []
    for variant_name, csv_path in sorted(variant_csv_map.items()):
        df = pd.read_csv(csv_path)

        # 保险处理：如果 CSV 里没有 variant_name，就补上
        if 'variant_name' not in df.columns:
            df['variant_name'] = variant_name

        all_dfs.append(df)

    merged = pd.concat(all_dfs, axis=0, ignore_index=True)
    return merged


def get_variant_label_and_color(variant_name):
    if variant_name in VARIANT_STYLE:
        return VARIANT_STYLE[variant_name]['label'], VARIANT_STYLE[variant_name]['color']
    return variant_name, None


# =========================================================
# 单实例图
# =========================================================
def plot_single_instance(df_all, instance_id, output_dir,
                         smooth_window=5, show_raw=True):
    df = df_all[df_all['instance_id'] == instance_id].copy()
    if len(df) == 0:
        print('[WARN] {} 没有数据，跳过'.format(instance_id))
        return None

    ensure_dir(output_dir)
    save_path = os.path.join(output_dir, '{}_benchmark_runs.png'.format(instance_id))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    variant_names = sorted(df['variant_name'].unique().tolist())

    for variant_name in variant_names:
        sub = df[df['variant_name'] == variant_name].copy()
        sub = sub.sort_values(by='run_id')

        x = sub['run_id'].values.astype(np.int32)
        y = sub['makespan'].values.astype(np.float64)
        y_smooth = moving_average(y, window=smooth_window)

        label, color = get_variant_label_and_color(variant_name)

        if show_raw:
            ax.plot(
                x, y,
                color=color,
                linewidth=RAW_LINEWIDTH,
                alpha=RAW_ALPHA
            )

        ax.plot(
            x, y_smooth,
            color=color,
            linewidth=SMOOTH_LINEWIDTH,
            label=label
        )

    ax.set_title(instance_id, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel('Run Index', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Completion Time', fontsize=LABEL_FONTSIZE)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('[OK] 已保存单实例图: {}'.format(save_path))
    return save_path


# =========================================================
# 多子图拼图
# =========================================================
def plot_multi_panel(df_all, instance_ids, output_dir,
                     smooth_window=5, show_raw=True,
                     ncols=2, filename='selected_instances_benchmark_runs.png'):
    valid_instances = []
    for instance_id in instance_ids:
        sub = df_all[df_all['instance_id'] == instance_id]
        if len(sub) > 0:
            valid_instances.append(instance_id)

    if len(valid_instances) == 0:
        print('[WARN] 没有可画的实例')
        return None

    ensure_dir(output_dir)

    n = len(valid_instances)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(float(n) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.6 * ncols, 4.6 * nrows)
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    flat_axes = axes.flatten()

    for idx, instance_id in enumerate(valid_instances):
        ax = flat_axes[idx]
        sub_df = df_all[df_all['instance_id'] == instance_id].copy()
        variant_names = sorted(sub_df['variant_name'].unique().tolist())

        for variant_name in variant_names:
            sub = sub_df[sub_df['variant_name'] == variant_name].copy()
            sub = sub.sort_values(by='run_id')

            x = sub['run_id'].values.astype(np.int32)
            y = sub['makespan'].values.astype(np.float64)
            y_smooth = moving_average(y, window=smooth_window)

            label, color = get_variant_label_and_color(variant_name)

            if show_raw:
                ax.plot(
                    x, y,
                    color=color,
                    linewidth=RAW_LINEWIDTH,
                    alpha=RAW_ALPHA
                )

            ax.plot(
                x, y_smooth,
                color=color,
                linewidth=SMOOTH_LINEWIDTH,
                label=label
            )

        ax.set_title(instance_id, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel('Run Index', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Completion Time', fontsize=LABEL_FONTSIZE)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)

    for idx in range(len(valid_instances), len(flat_axes)):
        flat_axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('[OK] 已保存拼图: {}'.format(save_path))
    return save_path


# =========================================================
# 汇总入口
# =========================================================
def plot_all_single_instances(df_all, instance_ids, output_dir,
                              smooth_window=5, show_raw=True):
    single_dir = os.path.join(output_dir, 'single_instances')
    ensure_dir(single_dir)

    saved = []
    for instance_id in instance_ids:
        path = plot_single_instance(
            df_all=df_all,
            instance_id=instance_id,
            output_dir=single_dir,
            smooth_window=smooth_window,
            show_raw=show_raw
        )
        if path is not None:
            saved.append(path)

    print('[DONE] 单实例图共生成 {} 张'.format(len(saved)))
    return saved


# =========================================================
# 参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--result_root',
        type=str,
        default=DEFAULT_RESULT_ROOT,
        help='ablation_suite 根目录'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='输出图目录'
    )

    parser.add_argument(
        '--instances',
        nargs='*',
        default=DEFAULT_INSTANCES,
        help='实例列表，例如 P1 P2 P10 P19'
    )

    parser.add_argument(
        '--smooth_window',
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help='平滑窗口'
    )

    parser.add_argument(
        '--hide_raw',
        action='store_true',
        help='不画原始细线，只画平滑线'
    )

    parser.add_argument(
        '--single_only',
        action='store_true',
        help='只画单实例图'
    )

    parser.add_argument(
        '--panel_only',
        action='store_true',
        help='只画拼图'
    )

    parser.add_argument(
        '--panel_ncols',
        type=int,
        default=2,
        help='拼图列数'
    )

    parser.add_argument(
        '--panel_filename',
        type=str,
        default='selected_instances_benchmark_runs.png',
        help='拼图文件名'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    show_raw = not args.hide_raw
    ensure_dir(args.output_dir)

    df_all = load_all_benchmark_runs(args.result_root)

    if not args.panel_only:
        plot_all_single_instances(
            df_all=df_all,
            instance_ids=args.instances,
            output_dir=args.output_dir,
            smooth_window=args.smooth_window,
            show_raw=show_raw
        )

    if not args.single_only:
        plot_multi_panel(
            df_all=df_all,
            instance_ids=args.instances,
            output_dir=args.output_dir,
            smooth_window=args.smooth_window,
            show_raw=show_raw,
            ncols=args.panel_ncols,
            filename=args.panel_filename
        )


if __name__ == '__main__':
    main()