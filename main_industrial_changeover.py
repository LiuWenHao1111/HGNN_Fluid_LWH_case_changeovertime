import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# 复用原 main.py 中的训练、验证、保存、推理逻辑
import main as base

from generate_data import load_industrial_orders
from mappo import PPO_Agent


# =========================================================
# 运行模式
# =========================================================
# train_industrial           : 重新训练工业数据模型
# eval_industrial_trained    : 只加载已训练工业模型并验证、重新生成甘特图
# train_and_eval_industrial  : 先训练，再验证
#
# 你现在已经训练过了，所以默认只验证和画图
# =========================================================
MODE = 'eval_industrial_trained'


# =========================================================
# 路径配置
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

INDUSTRIAL_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    'data',
    'instance_changeover'
)

INDUSTRIAL_RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    'result',
    'industrial_changeover_model'
)


# =========================================================
# 训练 / 验证配置
# =========================================================
INDUSTRIAL_TRAIN_ROUNDS = 480

INDUSTRIAL_EVAL_RUNS = 30

# True：保持原来随机采样验证逻辑，30 次取最优图
# False：贪心验证，图更稳定
INDUSTRIAL_STOCHASTIC_EVAL = True

VARIANT_NAME = 'industrial_changeover'

ABLATION_CFG = {
    'use_fluid_mask': True,
    'use_fluid_state': True,
    'use_fluid_reward_scale': True
}

INITIAL_LR = 0.0001
MIN_LR = 1e-5
LAST_K = 5
USE_VISDOM = True

RANDOM_SEED = 2026


# =========================================================
# 论文风格甘特图配置
# =========================================================
# 配色模式：
#   'type'    : 按工件类型 JC / TC / WC 配色，推荐论文图使用
#   'type_op' : 按 工件类型-工序 组合配色，颜色更多，不推荐论文图
GANTT_COLOR_MODE = 'type'

# 工件类型名称映射：
# 文中的 type 对应关系：
#   type 0 -> JC
#   type 1 -> TC
#   type 2 -> WC
TYPE_NAME_MAP = {
    0: 'JC',
    1: 'TC',
    2: 'WC'
}

# setup_time 统一颜色
SETUP_TIME_COLOR = '#4A4A4A'
SETUP_TIME_EDGE_COLOR = '#1F1F1F'

# setup_time 只作为窄条标记，不按真实切换时间拉宽
SETUP_TIME_MARKER_WIDTH_RATIO = 0.0028
SETUP_TIME_MARKER_MIN_WIDTH = 2.2
SETUP_TIME_MARKER_MAX_WIDTH = 4.8

# 加工条高度
BAR_HEIGHT = 0.62
SETUP_TIME_BAR_HEIGHT = 0.62

# 是否给工序块加标签
SHOW_OPERATION_LABEL = True

# 标注逻辑：
# 不再使用“每台机器最多标几个”的限制。
# 改为“根据文字长度和工序块宽度判断是否能放下”。
# 宽块一定会标注，短块放不下才不标。
LABEL_FONT_SIZE = 8.5
LABEL_FONT_WEIGHT = 'bold'

# 文字宽度估计参数：
# 数据坐标下，每个字符大概需要 makespan * LABEL_CHAR_WIDTH_RATIO 的宽度。
# 如果你发现标签还是偏多，可以调大；
# 如果你想标更多，可以调小。
LABEL_CHAR_WIDTH_RATIO = 0.0042

# 工序块至少要比估计文字宽度多一点，防止文字贴边
LABEL_WIDTH_PADDING_RATIO = 1.15

# 低于这个时长的块一般不标，避免极短块乱
LABEL_ABSOLUTE_MIN_DURATION = 18.0

# 短标签策略：
# full label: JC-O7
# short label: O7
# 如果 full label 放不下，但 short label 放得下，则显示 short label。
ALLOW_SHORT_LABEL_FALLBACK = True

# 只保存 PNG。
# PDF / SVG 保存代码在 _save_vector_versions() 中保留但注释。
SAVE_VECTOR_FIGURE = False

# 图片分辨率
FIGURE_DPI = 400


# 色盲友好 / 论文友好配色
# 当前 GANTT_COLOR_MODE='type'，所以主要使用前三个颜色：
# JC: 蓝色，TC: 橙色，WC: 绿色
TYPE_PALETTE = [
    '#4E79A7',  # JC - blue
    '#F28E2B',  # TC - orange
    '#59A14F',  # WC - green
    '#E15759',  # red
    '#B07AA1',  # purple
    '#9C755F',  # brown
    '#76B7B2',  # teal
    '#EDC948',  # yellow
    '#AF7AA1',  # violet
    '#FF9DA7',  # pink
]

TYPE_OP_PALETTE = [
    '#4E79A7', '#A0CBE8',
    '#F28E2B', '#FFBE7D',
    '#59A14F', '#8CD17D',
    '#B6992D', '#F1CE63',
    '#499894', '#86BCB6',
    '#E15759', '#FF9D9A',
    '#79706E', '#BAB0AC',
    '#D37295', '#FABFD2',
    '#B07AA1', '#D4A6C8',
    '#9D7660', '#D7B5A6',
]


# =========================================================
# 全局基础设置
# =========================================================
def set_global_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_publication_matplotlib_style():
    """
    不依赖 seaborn，直接用 matplotlib 设置论文风格。
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['figure.dpi'] = FIGURE_DPI
    plt.rcParams['savefig.dpi'] = FIGURE_DPI

    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9

    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_type_display_name(type_id):
    """
    将内部 type_id 映射为论文中的工件类型名称。
    例如：
      0 -> JC
      1 -> TC
      2 -> WC
    """
    type_id = int(type_id)
    return TYPE_NAME_MAP.get(type_id, 'Type {}'.format(type_id))


def get_edge_dim_from_instance(data):
    """
    工业实例启用 setup_time / changeover 弧特征时 edge_dim = 7；
    普通实例仍为 edge_dim = 6。
    """
    if bool(data.get('is_industrial_instance', False)) and bool(data.get('use_changeover_feature', False)):
        return 7
    return 6


def build_agent(initial_lr=0.0001, edge_dim=None):
    """
    edge_dim 默认给 7，是为了防止原 main.py 的训练函数调用 build_agent()
    时没有传 edge_dim。
    当前脚本只处理工业 setup_time/changeover 模型，所以默认 7。
    """
    if edge_dim is None:
        edge_dim = 7

    agent = PPO_Agent(
        op_dim=8,
        m_dim=7,
        edge_dim=int(edge_dim),
        lr=initial_lr,
        batch_size=256
    )
    return agent


def load_policy_state_dict_compatible(agent, state_dict):
    """
    兼容 edge_dim=6 的旧模型和 edge_dim=7 的 setup_time/changeover 模型。
    你现在重新训练后的工业模型一般会 strict 加载。
    """
    try:
        agent.policy.load_state_dict(state_dict)
        return 'strict'

    except RuntimeError as err:
        current_state = agent.policy.state_dict()
        adapted_state = {}
        skipped = []

        for name, target_tensor in current_state.items():
            if name not in state_dict:
                adapted_state[name] = target_tensor
                skipped.append((name, 'missing', tuple(target_tensor.shape)))
                continue

            source_tensor = state_dict[name]

            if tuple(source_tensor.shape) == tuple(target_tensor.shape):
                adapted_state[name] = source_tensor
                continue

            if (
                source_tensor.ndim == 2 and
                target_tensor.ndim == 2 and
                source_tensor.shape[0] == target_tensor.shape[0] and
                source_tensor.shape[1] < target_tensor.shape[1]
            ):
                new_tensor = target_tensor.clone()
                new_tensor[:, :source_tensor.shape[1]] = source_tensor
                adapted_state[name] = new_tensor
                continue

            adapted_state[name] = target_tensor
            skipped.append((name, tuple(source_tensor.shape), tuple(target_tensor.shape)))

        agent.policy.load_state_dict(adapted_state, strict=True)

        print('⚠️ 模型结构与当前 edge_dim 不完全一致，已做兼容加载。')
        print('   原始错误: {}'.format(str(err).split('\n')[0]))
        if len(skipped) > 0:
            print('   保持当前初始化的参数数: {}'.format(len(skipped)))

        return 'adapted'


# =========================================================
# 调度明细导出：保持原格式，额外保存 setup_time 字段
# =========================================================
def export_schedule_detail_with_setup_time(schedule_log,
                                           save_dir='result',
                                           filename='best_schedule_detail.csv',
                                           machine_labels=None,
                                           original_machine_labels=None,
                                           compact_to_original_machine_id=None):
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, filename)

    rows = []
    schedule_sorted = sorted(schedule_log, key=lambda x: (x['machine'], x['start'], x['end']))

    for idx, item in enumerate(schedule_sorted):
        compact_machine = int(item['machine'])

        row = {
            'seq': idx + 1,
            'compact_machine_id': compact_machine + 1,
            'compact_machine_label': machine_labels[compact_machine]
            if machine_labels is not None and compact_machine < len(machine_labels)
            else 'M{}'.format(compact_machine + 1),
            'type': int(item['type']),
            'type_name': get_type_display_name(int(item['type'])),
            'op': int(item['op']),
            'start': float(item['start']),
            'end': float(item['end']),
            'duration': float(item['end'] - item['start'])
        }

        if compact_to_original_machine_id is not None and compact_machine < len(compact_to_original_machine_id):
            old_m = int(compact_to_original_machine_id[compact_machine])
            row['original_machine_id'] = old_m + 1

            if original_machine_labels is not None and compact_machine < len(original_machine_labels):
                row['original_machine_label'] = original_machine_labels[compact_machine]
            else:
                row['original_machine_label'] = 'M{}'.format(old_m + 1)

        if 'job_idx_in_type' in item:
            row['job_idx_in_type'] = int(item['job_idx_in_type'])

        if 'original_type' in item:
            row['original_type'] = int(item['original_type'])
            row['original_type_name'] = get_type_display_name(int(item['original_type']))

        # env.py 里字段名仍然可能叫 changeover_time_feature。
        # 这里导出时改成论文中的 setup_time_feature。
        if 'changeover_time_feature' in item:
            row['setup_time_feature'] = float(item['changeover_time_feature'])
        elif 'setup_time_feature' in item:
            row['setup_time_feature'] = float(item['setup_time_feature'])

        rows.append(row)

    base.save_csv_rows(rows, save_path)
    return save_path


# =========================================================
# 甘特图配色与标注工具
# =========================================================
def _get_processing_color_key(entry):
    r = int(entry['type'])
    j = int(entry['op'])

    if GANTT_COLOR_MODE == 'type':
        return ('type', r)

    return ('type_op', r, j)


def _build_processing_color_map(schedule_log):
    keys = []
    for entry in schedule_log:
        key = _get_processing_color_key(entry)
        if key not in keys:
            keys.append(key)

    keys = sorted(keys)

    palette = TYPE_PALETTE if GANTT_COLOR_MODE == 'type' else TYPE_OP_PALETTE

    color_map = {}
    for i, key in enumerate(keys):
        color_map[key] = palette[i % len(palette)]

    return color_map


def _get_text_color_for_fill(color_hex):
    """
    根据背景色亮度自动选择黑字或白字。
    """
    try:
        r, g, b = mcolors.to_rgb(color_hex)
    except Exception:
        return 'white'

    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black' if luminance > 0.62 else 'white'


def _get_setup_time_marker_width(setup_value, duration, makespan):
    """
    setup_time 作为窄条标记，不按真实切换时间画宽度。
    """
    duration = max(float(duration), 1e-9)
    makespan = max(float(makespan), 1.0)

    marker_width = makespan * SETUP_TIME_MARKER_WIDTH_RATIO
    marker_width = max(marker_width, SETUP_TIME_MARKER_MIN_WIDTH)
    marker_width = min(marker_width, SETUP_TIME_MARKER_MAX_WIDTH)
    marker_width = min(marker_width, duration)

    return marker_width


def _make_operation_label(entry, short=False):
    """
    工序块内部标签。

    full:
        JC-O7 / TC-O1 / WC-O3

    short:
        O7 / O1 / O3
    """
    r = int(entry['type'])
    j = int(entry['op'])

    if short:
        return 'O{}'.format(j)

    type_name = get_type_display_name(r)
    return '{}-O{}'.format(type_name, j)


def _estimate_label_width_data(label, makespan):
    """
    用数据坐标粗略估计标签需要的横向宽度。
    这样可以避免字超出工序块，同时不再需要每台机器标签数量上限。

    估计逻辑：
      label 字符数 * makespan * LABEL_CHAR_WIDTH_RATIO
    """
    label_len = max(1, len(str(label)))
    return float(label_len) * float(makespan) * LABEL_CHAR_WIDTH_RATIO


def _choose_label_for_block(entry, duration, makespan):
    """
    根据工序块宽度选择标签：
    1. 如果 full label 能放下，显示 full label；
    2. 如果 full label 放不下，但 short label 能放下，显示 short label；
    3. 如果都放不下，不显示标签。
    """
    duration = float(duration)
    makespan = max(float(makespan), 1.0)

    if duration < LABEL_ABSOLUTE_MIN_DURATION:
        return None

    full_label = _make_operation_label(entry, short=False)
    full_label_width = _estimate_label_width_data(full_label, makespan) * LABEL_WIDTH_PADDING_RATIO

    if duration >= full_label_width:
        return full_label

    if ALLOW_SHORT_LABEL_FALLBACK:
        short_label = _make_operation_label(entry, short=True)
        short_label_width = _estimate_label_width_data(short_label, makespan) * LABEL_WIDTH_PADDING_RATIO

        if duration >= short_label_width:
            return short_label

    return None


def _save_vector_versions(fig, save_path):
    """
    现在只保存 PNG。
    PDF/SVG 代码按你的要求注释掉，不删除。
    """
    return

    # 如果后续投稿需要矢量图，可以取消下面三行注释：
    # root, _ = os.path.splitext(save_path)
    # fig.savefig(root + '.pdf', bbox_inches='tight')
    # fig.savefig(root + '.svg', bbox_inches='tight')


def _get_setup_time_value(entry):
    """
    兼容两种字段名：
    - changeover_time_feature：env.py 里原来的字段名
    - setup_time_feature：如果你后面把 env.py 也改名了，也能兼容
    """
    if 'setup_time_feature' in entry:
        return float(entry.get('setup_time_feature', 0.0))
    return float(entry.get('changeover_time_feature', 0.0))


# =========================================================
# 论文风格甘特图
# =========================================================
def save_gantt_chart_publication(schedule_log,
                                 num_machines,
                                 makespan,
                                 save_dir='result',
                                 filename='best_gantt.png',
                                 machine_labels=None,
                                 drop_idle_machines=False):
    """
    论文风格甘特图：
    1. 加工块按 JC / TC / WC 配色；
    2. setup_time 用统一深灰窄条表示；
    3. 不显示具体 setup_time 数值；
    4. 标签不再设置每台机器数量上限，而是根据工序块宽度自动判断；
    5. 宽工序块一定会优先完整标注；
    6. 短工序块如果放不下 full label，会尝试短标签 O_j；
    7. 仍放不下则不标，避免图乱；
    8. 只保存 PNG，PDF/SVG 代码已注释。
    """
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, filename)

    if len(schedule_log) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title('Empty Schedule')
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches='tight')
        _save_vector_versions(fig, save_path)
        plt.close(fig)
        return save_path

    if drop_idle_machines:
        machine_list = sorted(list(set(int(e['machine']) for e in schedule_log)))
    else:
        machine_list = list(range(int(num_machines)))

    machine_pos = {}
    for idx, m in enumerate(machine_list):
        machine_pos[m] = idx

    num_rows = max(1, len(machine_list))
    fig_height = max(5.2, 0.34 * num_rows + 1.8)
    fig_width = 14.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    color_map = _build_processing_color_map(schedule_log)

    schedule_sorted = sorted(
        schedule_log,
        key=lambda x: (int(x['machine']), float(x['start']), float(x['end']))
    )

    for entry in schedule_sorted:
        m = int(entry['machine'])

        if m not in machine_pos:
            continue

        start = float(entry['start'])
        end = float(entry['end'])
        duration = max(0.0, end - start)

        if duration <= 1e-9:
            continue

        y = machine_pos[m]

        color_key = _get_processing_color_key(entry)
        fill_color = color_map.get(color_key, '#4E79A7')

        # 1) 加工块
        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=BAR_HEIGHT,
            color=fill_color,
            edgecolor='#2F2F2F',
            linewidth=0.65,
            alpha=0.95,
            zorder=2
        )

        # 2) setup_time 窄条：统一深灰，不标数字
        setup_value = _get_setup_time_value(entry)

        if setup_value > 1e-9:
            setup_width = _get_setup_time_marker_width(
                setup_value=setup_value,
                duration=duration,
                makespan=makespan
            )

            ax.barh(
                y=y,
                width=setup_width,
                left=start,
                height=SETUP_TIME_BAR_HEIGHT,
                color=SETUP_TIME_COLOR,
                edgecolor=SETUP_TIME_EDGE_COLOR,
                linewidth=0.55,
                alpha=1.0,
                zorder=4
            )

        # 3) 标签：
        # 不再按每台机器数量限制。
        # 根据块宽和文字宽度判断是否能放下。
        if SHOW_OPERATION_LABEL:
            label = _choose_label_for_block(
                entry=entry,
                duration=duration,
                makespan=makespan
            )

            if label is not None:
                text_color = _get_text_color_for_fill(fill_color)

                ax.text(
                    start + duration / 2.0,
                    y,
                    label,
                    ha='center',
                    va='center',
                    fontsize=LABEL_FONT_SIZE,
                    fontweight=LABEL_FONT_WEIGHT,
                    color=text_color,
                    zorder=5,
                    clip_on=True
                )

    # y 轴
    ax.set_yticks(range(len(machine_list)))

    if machine_labels is not None:
        y_labels = []
        for m in machine_list:
            if 0 <= m < len(machine_labels):
                y_labels.append(machine_labels[m])
            else:
                y_labels.append('M{}'.format(m + 1))
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticklabels(['M{}'.format(m + 1) for m in machine_list])

    # 坐标轴和标题
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Best Schedule Gantt Chart (Makespan: {:.2f})'.format(float(makespan)), pad=10)

    # 网格只保留 x 方向，线条更淡
    ax.grid(axis='x', linestyle='--', linewidth=0.55, alpha=0.28)
    ax.grid(axis='y', visible=False)

    # 坐标范围留白
    ax.set_xlim(0, float(makespan) * 1.04)
    ax.set_ylim(-0.75, len(machine_list) - 0.25)

    # 机器编号从上到下显示。
    # 如果你想 M1 在最下面，可以注释掉这一行。
    ax.invert_yaxis()

    # 图例
    legend_handles = []

    if GANTT_COLOR_MODE == 'type':
        used_types = sorted(list(set(int(e['type']) for e in schedule_log)))
        for t in used_types:
            key = ('type', t)
            if key in color_map:
                legend_handles.append(
                    Patch(
                        facecolor=color_map[key],
                        edgecolor='#2F2F2F',
                        label=get_type_display_name(t)
                    )
                )

    legend_handles.append(
        Patch(
            facecolor=SETUP_TIME_COLOR,
            edgecolor=SETUP_TIME_EDGE_COLOR,
            label='setup_time'
        )
    )

    ax.legend(
        handles=legend_handles,
        loc='upper right',
        frameon=True,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        ncol=1
    )

    # 边框简洁化
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    plt.tight_layout()

    # 只保存 PNG
    fig.savefig(save_path, bbox_inches='tight')

    # PDF / SVG 保存已注释
    _save_vector_versions(fig, save_path)

    plt.close(fig)

    return save_path


# =========================================================
# 覆盖 base 中的函数
# 保持原本训练 / 验证主逻辑不变，只替换画图与兼容部分
# =========================================================
base.build_agent = build_agent
base.get_edge_dim_from_instance = get_edge_dim_from_instance
base.load_policy_state_dict_compatible = load_policy_state_dict_compatible
base.export_schedule_detail = export_schedule_detail_with_setup_time
base.save_gantt_chart = save_gantt_chart_publication

base.INITIAL_LR = INITIAL_LR
base.MIN_LR = MIN_LR
base.LAST_K = LAST_K
base.USE_VISDOM = USE_VISDOM
base.EVAL_RUNS = INDUSTRIAL_EVAL_RUNS
base.STOCHASTIC_EVAL = INDUSTRIAL_STOCHASTIC_EVAL


# =========================================================
# 数据读取
# =========================================================
def load_industrial_instance_list():
    print('\n[DEBUG] 当前工作目录 cwd = {}'.format(os.getcwd()))
    print('[DEBUG] 工业数据目录 INDUSTRIAL_DATA_DIR = {}'.format(INDUSTRIAL_DATA_DIR))
    print('[DEBUG] 目录是否存在 = {}'.format(os.path.exists(INDUSTRIAL_DATA_DIR)))

    if os.path.exists(INDUSTRIAL_DATA_DIR):
        print('[DEBUG] 目录下文件 = {}'.format(os.listdir(INDUSTRIAL_DATA_DIR)))

    order_instances = load_industrial_orders(
        instance_dir=INDUSTRIAL_DATA_DIR,
        compact_unused_machines=True,
        drop_zero_job_types=True
    )

    if len(order_instances) == 0:
        raise ValueError('没有读取到工业订单实例，请检查目录: {}'.format(INDUSTRIAL_DATA_DIR))

    return order_instances


# =========================================================
# 重新训练工业模型
# 默认 MODE 不会调用，保留完整逻辑
# =========================================================
def train_industrial_model():
    set_global_seed(RANDOM_SEED)
    set_publication_matplotlib_style()

    order_instances = load_industrial_instance_list()
    max_epochs = int(len(order_instances) * INDUSTRIAL_TRAIN_ROUNDS)

    first_edge_dim = get_edge_dim_from_instance(order_instances[0][1])

    print('\n==============================================================')
    print('重新训练工业数据 setup_time 模型')
    print('工业订单数       = {}'.format(len(order_instances)))
    print('训练轮数         = {}'.format(INDUSTRIAL_TRAIN_ROUNDS))
    print('总 occurrence    = {}'.format(max_epochs))
    print('edge_dim         = {}'.format(first_edge_dim))
    print('结果目录         = {}'.format(INDUSTRIAL_RESULT_DIR))
    print('==============================================================')

    base.train_round_robin_on_instance_list(
        instance_list=order_instances,
        max_epochs=max_epochs,
        result_dir=INDUSTRIAL_RESULT_DIR,
        variant_name=VARIANT_NAME,
        ablation_cfg=ABLATION_CFG,
        visdom_env_name='industrial_setup_time_train',
        curve_window_last_k=base.LAST_K
    )


# =========================================================
# 加载已训练工业模型，重新验证并生成论文风格甘特图
# =========================================================
def evaluate_trained_industrial_model():
    set_global_seed(RANDOM_SEED)
    set_publication_matplotlib_style()

    order_instances = load_industrial_instance_list()

    model_path = os.path.join(INDUSTRIAL_RESULT_DIR, 'best_model.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError('未找到工业数据训练后的 best_model.pth: {}'.format(model_path))

    out_dir = os.path.join(INDUSTRIAL_RESULT_DIR, 'industrial_eval')
    per_order_dir = os.path.join(out_dir, 'per_order')
    summary_csv = os.path.join(out_dir, 'industrial_table16_style_summary.csv')
    all_runs_csv = os.path.join(out_dir, 'industrial_all_runs.csv')

    ensure_dir(out_dir)
    ensure_dir(per_order_dir)

    industrial_edge_dim = get_edge_dim_from_instance(order_instances[0][1])

    agent = build_agent(
        initial_lr=INITIAL_LR,
        edge_dim=industrial_edge_dim
    )

    state_dict = torch.load(model_path, map_location=agent.device)
    load_policy_state_dict_compatible(agent, state_dict)
    agent.policy.eval()

    all_run_rows = []
    summary_rows = []

    print('\n🚀 开始加载已训练工业模型，并重新生成论文风格甘特图')
    print('model_path = {}'.format(model_path))
    print('edge_dim   = {}'.format(industrial_edge_dim))
    print('color_mode = {}'.format(GANTT_COLOR_MODE))
    print('stochastic = {}'.format(INDUSTRIAL_STOCHASTIC_EVAL))
    print('runs       = {}'.format(INDUSTRIAL_EVAL_RUNS))

    for order_name, order_data in order_instances:
        order_folder = os.path.join(per_order_dir, order_name)
        ensure_dir(order_folder)

        counts = order_data.get('order_kind_counts', [])

        print('\n{}:'.format(order_name))
        print('  kind_counts             = {}'.format(counts))
        print('  active_job_types        = {}'.format(order_data['num_job_types']))
        print('  active_machines         = {}'.format(order_data['num_machines']))
        print('  compact_machine_labels  = {}'.format(order_data.get('machine_labels', [])))
        print('  original_machine_labels = {}'.format(order_data.get('original_machine_labels', [])))
        print('  edge_dim                = {}'.format(get_edge_dim_from_instance(order_data)))
        print('  setup_time_feature      = {}'.format(bool(order_data.get('use_changeover_feature', False))))
        print('  has_setup_time_data     = {}'.format(bool(order_data.get('has_changeover_data', False))))

        machine_mapping_csv = base.save_machine_mapping_csv(
            machine_labels=order_data.get('machine_labels', []),
            original_machine_labels=order_data.get('original_machine_labels', []),
            compact_to_original_machine_id=order_data.get('compact_to_original_machine_id', []),
            save_dir=order_folder,
            filename='machine_mapping.csv'
        )

        cmax_list = []
        tcpu_list = []
        best_cmax = float('inf')
        best_res = None
        order_run_rows = []

        for run_id in range(1, INDUSTRIAL_EVAL_RUNS + 1):
            res = base.run_policy_inference_once(
                agent=agent,
                instance_data=order_data,
                ablation_cfg=ABLATION_CFG,
                stochastic=INDUSTRIAL_STOCHASTIC_EVAL
            )

            cmax = float(res['makespan'])
            tcpu = float(res['cpu_time'])

            cmax_list.append(cmax)
            tcpu_list.append(tcpu)

            row = {
                'variant_name': VARIANT_NAME,
                'order': order_name,
                'run_id': int(run_id),
                'eval_mode': 'stochastic' if INDUSTRIAL_STOCHASTIC_EVAL else 'greedy',
                'edge_dim': int(industrial_edge_dim),
                'setup_time_feature': bool(industrial_edge_dim == 7),
                'makespan': cmax,
                'cpu_time_sec': tcpu
            }

            order_run_rows.append(row)
            all_run_rows.append(row)

            if cmax < best_cmax:
                best_cmax = cmax
                best_res = copy.deepcopy(res)

            print('  run {:02d}/{} | Cmax = {:.2f} | Tcpu = {:.4f}s'.format(
                run_id,
                INDUSTRIAL_EVAL_RUNS,
                cmax,
                tcpu
            ))

        cb = float(np.min(cmax_list))
        ca = float(np.mean(cmax_list))
        tcpu_avg = float(np.mean(tcpu_list))

        order_runs_csv = os.path.join(order_folder, 'all_runs.csv')
        base.save_csv_rows(order_run_rows, order_runs_csv)

        best_schedule_csv = export_schedule_detail_with_setup_time(
            best_res['schedule_log'],
            save_dir=order_folder,
            filename='best_schedule_detail.csv',
            machine_labels=best_res['machine_labels'],
            original_machine_labels=best_res['original_machine_labels'],
            compact_to_original_machine_id=best_res['compact_to_original_machine_id']
        )

        best_gantt_full_active = save_gantt_chart_publication(
            best_res['schedule_log'],
            best_res['num_machines'],
            best_res['makespan'],
            save_dir=order_folder,
            filename='best_gantt_full_active.png',
            machine_labels=best_res['machine_labels'],
            drop_idle_machines=False
        )

        best_gantt_used_only = save_gantt_chart_publication(
            best_res['schedule_log'],
            best_res['num_machines'],
            best_res['makespan'],
            save_dir=order_folder,
            filename='best_gantt_used_only.png',
            machine_labels=best_res['machine_labels'],
            drop_idle_machines=True
        )

        summary_row = {
            'variant_name': VARIANT_NAME,
            'order': order_name,
            'runs': int(INDUSTRIAL_EVAL_RUNS),
            'eval_mode': 'stochastic' if INDUSTRIAL_STOCHASTIC_EVAL else 'greedy',
            'edge_dim': int(industrial_edge_dim),
            'setup_time_feature': bool(industrial_edge_dim == 7),
            'Cb': cb,
            'Ca': ca,
            'Tcpu_avg_sec': tcpu_avg,
            'best_schedule_csv': best_schedule_csv,
            'machine_mapping_csv': machine_mapping_csv,
            'best_gantt_full_active': best_gantt_full_active,
            'best_gantt_used_only': best_gantt_used_only
        }

        for k, v in enumerate(counts):
            summary_row['kind{}_count'.format(k)] = int(v)

        summary_rows.append(summary_row)

        print('  [SUMMARY] Cb = {:.2f}, Ca = {:.2f}, Tcpu = {:.4f}s'.format(
            cb,
            ca,
            tcpu_avg
        ))
        print('  甘特图 full active = {}'.format(best_gantt_full_active))
        print('  甘特图 used only   = {}'.format(best_gantt_used_only))
        print('  当前版本只保存 PNG 图。PDF / SVG 保存代码已注释。')

    base.save_csv_rows(all_run_rows, all_runs_csv)
    base.save_csv_rows(summary_rows, summary_csv)

    print('\n✅ 已完成：加载训练好的工业模型，并重新生成论文风格甘特图')
    print('全部运行明细: {}'.format(all_runs_csv))
    print('汇总结果    : {}'.format(summary_csv))


# =========================================================
# main
# =========================================================
def main():
    if MODE == 'train_industrial':
        train_industrial_model()

    elif MODE == 'eval_industrial_trained':
        evaluate_trained_industrial_model()

    elif MODE == 'train_and_eval_industrial':
        train_industrial_model()
        evaluate_trained_industrial_model()

    else:
        raise ValueError('未知 MODE: {}'.format(MODE))


if __name__ == '__main__':
    main()