import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 路径配置（按你当前目录结构写好的）
# =========================================================
PROJECT_ROOT = '.'

# baseline 单独训练结果目录
BASELINE_ROOT = os.path.join(PROJECT_ROOT, 'result', 'round_robin_24cases')

# 消融实验目录
ABLATION_ROOT = os.path.join(PROJECT_ROOT, 'result', 'ablation_suite')

# 输出目录
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'result', 'benchmark_eval_plots')

# 你当前有的几个变体
VARIANT_CONFIGS = [
    {
        'name': 'baseline',
        'label': 'FINDRL',
        'history_root': os.path.join(BASELINE_ROOT, 'per_instance'),
        'color': '#1f77b4'   # 蓝
    },
    {
        'name': 'ablation_no_zero_mask',
        'label': 'FINDRL-NM',
        'history_root': os.path.join(ABLATION_ROOT, 'ablation_no_zero_mask', 'per_instance'),
        'color': '#d62728'   # 红
    },
    {
        'name': 'ablation_no_fluid_state',
        'label': 'FINDRL-NS',
        'history_root': os.path.join(ABLATION_ROOT, 'ablation_no_fluid_state', 'per_instance'),
        'color': '#2ca02c'   # 绿
    },
    {
        'name': 'ablation_no_reward_scale',
        'label': 'FINDRL-NR',
        'history_root': os.path.join(ABLATION_ROOT, 'ablation_no_reward_scale', 'per_instance'),
        'color': '#ff7f0e'   # 橙
    },
]

# 默认挑几张先画成一页，便于挑论文图
DEFAULT_SELECTED_INSTANCES = ['P1', 'P2', 'P10', 'P19']

# 训练曲线字段
X_FIELD = 'instance_occurrence'
Y_FIELD = 'current_objective'

# 平滑窗口
SMOOTH_WINDOW = 5

# 画图参数
PLOT_RAW_LINE = True
PLOT_SMOOTH_LINE = True
RAW_ALPHA = 0.20
RAW_LINEWIDTH = 0.8
SMOOTH_LINEWIDTH = 2.2

SINGLE_FIGSIZE = (7, 5)
GRID_FIGSIZE = (14, 10)


# =========================================================
# 基础工具
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_csv_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv_rows(rows, save_path, fieldnames=None):
    ensure_dir(os.path.dirname(save_path))
    if len(rows) == 0:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(y, window=5):
    if len(y) == 0:
        return np.array([])
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros(len(y), dtype=np.float64)
    for i in range(len(y)):
        start = max(0, i - window + 1)
        out[i] = np.mean(y[start:i + 1])
    return out


def parse_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def parse_int(v, default=None):
    try:
        return int(float(v))
    except Exception:
        return default


def get_history_csv_path(history_root, instance_id):
    return os.path.join(history_root, instance_id, 'history.csv')


def load_curve_data(history_root, instance_id):
    history_csv = get_history_csv_path(history_root, instance_id)
    rows = read_csv_rows(history_csv)

    x_vals = []
    y_vals = []

    for idx, row in enumerate(rows):
        x = parse_int(row.get(X_FIELD), None)
        y = parse_float(row.get(Y_FIELD), None)

        if x is None:
            x = idx + 1
        if y is None:
            continue

        x_vals.append(x)
        y_vals.append(y)

    return x_vals, y_vals, history_csv


def save_figure(fig, save_path):
    ensure_dir(os.path.dirname(save_path))
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# 数据索引导出
# =========================================================
def export_training_data_index():
    rows = []
    all_instances = ['P{}'.format(i) for i in range(1, 25)]

    for cfg in VARIANT_CONFIGS:
        for instance_id in all_instances:
            history_csv = get_history_csv_path(cfg['history_root'], instance_id)
            rows.append({
                'variant_name': cfg['name'],
                'variant_label': cfg['label'],
                'instance_id': instance_id,
                'history_csv_path': history_csv,
                'exists': int(os.path.exists(history_csv))
            })

    save_path = os.path.join(OUTPUT_ROOT, 'training_curve_data_index.csv')
    save_csv_rows(
        rows,
        save_path,
        fieldnames=['variant_name', 'variant_label', 'instance_id', 'history_csv_path', 'exists']
    )
    return save_path


# =========================================================
# 单算例图
# =========================================================
def plot_single_instance_compare(instance_id):
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    has_any_data = False

    for cfg in VARIANT_CONFIGS:
        x_vals, y_vals, history_csv = load_curve_data(cfg['history_root'], instance_id)

        if len(x_vals) == 0:
            print('[WARN] 找不到训练数据: {} | {} | {}'.format(instance_id, cfg['name'], history_csv))
            continue

        has_any_data = True
        y_smooth = moving_average(y_vals, window=SMOOTH_WINDOW)

        if PLOT_RAW_LINE:
            ax.plot(
                x_vals,
                y_vals,
                color=cfg['color'],
                alpha=RAW_ALPHA,
                linewidth=RAW_LINEWIDTH
            )

        if PLOT_SMOOTH_LINE:
            ax.plot(
                x_vals,
                y_smooth,
                color=cfg['color'],
                linewidth=SMOOTH_LINEWIDTH,
                label=cfg['label']
            )

    if not has_any_data:
        plt.close(fig)
        return None

    ax.set_title(instance_id, fontsize=13)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Completion Time', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=9, loc='best')

    save_path = os.path.join(OUTPUT_ROOT, 'single_instance_figures', '{}_compare.png'.format(instance_id))
    save_figure(fig, save_path)
    return save_path


# =========================================================
# 多子图
# =========================================================
def plot_instance_grid(instance_ids, filename):
    n = len(instance_ids)
    if n == 0:
        return None

    ncols = 2
    nrows = int(math.ceil(float(n) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=GRID_FIGSIZE)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes_flat = axes.flatten()

    for idx, instance_id in enumerate(instance_ids):
        ax = axes_flat[idx]
        has_any_data = False

        for cfg in VARIANT_CONFIGS:
            x_vals, y_vals, history_csv = load_curve_data(cfg['history_root'], instance_id)

            if len(x_vals) == 0:
                print('[WARN] 找不到训练数据: {} | {} | {}'.format(instance_id, cfg['name'], history_csv))
                continue

            has_any_data = True
            y_smooth = moving_average(y_vals, window=SMOOTH_WINDOW)

            if PLOT_RAW_LINE:
                ax.plot(
                    x_vals,
                    y_vals,
                    color=cfg['color'],
                    alpha=RAW_ALPHA,
                    linewidth=RAW_LINEWIDTH
                )

            if PLOT_SMOOTH_LINE:
                ax.plot(
                    x_vals,
                    y_smooth,
                    color=cfg['color'],
                    linewidth=SMOOTH_LINEWIDTH,
                    label=cfg['label']
                )

        ax.set_title(instance_id, fontsize=12)
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Completion Time', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.35)
        if has_any_data:
            ax.legend(fontsize=8, loc='best')

    # 多余子图关掉
    for j in range(len(instance_ids), len(axes_flat)):
        axes_flat[j].axis('off')

    save_path = os.path.join(OUTPUT_ROOT, filename)
    save_figure(fig, save_path)
    return save_path


# =========================================================
# 批量画全部
# =========================================================
def plot_all_instances_separately():
    saved_paths = []
    for i in range(1, 25):
        instance_id = 'P{}'.format(i)
        save_path = plot_single_instance_compare(instance_id)
        if save_path is not None:
            saved_paths.append(save_path)
    return saved_paths


def plot_all_instances_in_pages(page_size=4):
    saved_paths = []
    all_instances = ['P{}'.format(i) for i in range(1, 25)]

    for page_idx in range(0, len(all_instances), page_size):
        sub_instances = all_instances[page_idx: page_idx + page_size]
        filename = os.path.join(
            'paged_figures',
            'all_instances_page_{:02d}.png'.format(page_idx // page_size + 1)
        )
        save_path = plot_instance_grid(sub_instances, filename)
        if save_path is not None:
            saved_paths.append(save_path)

    return saved_paths


# =========================================================
# 输出一个总说明
# =========================================================
def export_summary_note():
    rows = []
    for cfg in VARIANT_CONFIGS:
        rows.append({
            'variant_name': cfg['name'],
            'variant_label': cfg['label'],
            'history_root': cfg['history_root'],
            'exists': int(os.path.exists(cfg['history_root']))
        })

    save_path = os.path.join(OUTPUT_ROOT, 'curve_source_roots.csv')
    save_csv_rows(
        rows,
        save_path,
        fieldnames=['variant_name', 'variant_label', 'history_root', 'exists']
    )
    return save_path


# =========================================================
# main
# =========================================================
def main():
    ensure_dir(OUTPUT_ROOT)

    print('====================================================')
    print('开始绘制训练阶段消融对比曲线')
    print('BASELINE_ROOT = {}'.format(BASELINE_ROOT))
    print('ABLATION_ROOT = {}'.format(ABLATION_ROOT))
    print('OUTPUT_ROOT   = {}'.format(OUTPUT_ROOT))
    print('====================================================')

    source_note = export_summary_note()
    print('曲线源目录索引已保存: {}'.format(source_note))

    index_csv = export_training_data_index()
    print('训练数据索引已保存: {}'.format(index_csv))

    selected_path = plot_instance_grid(
        DEFAULT_SELECTED_INSTANCES,
        'selected_instances_compare.png'
    )
    print('已保存精选多子图: {}'.format(selected_path))

    single_paths = plot_all_instances_separately()
    print('已保存单算例图数量: {}'.format(len(single_paths)))
    print('目录: {}'.format(os.path.join(OUTPUT_ROOT, 'single_instance_figures')))

    page_paths = plot_all_instances_in_pages(page_size=4)
    print('已保存分页图数量: {}'.format(len(page_paths)))
    print('目录: {}'.format(os.path.join(OUTPUT_ROOT, 'paged_figures')))

    print('全部完成。')


if __name__ == '__main__':
    main()