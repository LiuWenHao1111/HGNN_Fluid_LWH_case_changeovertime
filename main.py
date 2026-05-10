import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
import time
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mappo import PPO_Agent
from generate_data import load_all_standard_instances, load_industrial_orders, load_dslfmae_instances
from env_ablation import AblationFJSPEnv

try:
    from visdom import Visdom
except Exception:
    Visdom = None


# =========================================================
# 运行模式
# 旧逻辑（保留）：
# train_round_robin
# eval_best_model
# eval_industrial
# train_and_eval
#
# 消融逻辑（保留）：
# train_single_variant
# eval_single_variant
# train_and_eval_single
# run_public_ablation_suite
#
# DSLFMAE 训练（保留）：
# run_dslfmae_table13
#
# 新增 DSLFMAE 已训练模型评测：
# eval_dslfmae_table13
# =========================================================
MODE = 'eval_dslfmae_table13'

# 单变体模式时用这个名字
SINGLE_VARIANT_NAME = 'baseline'

# 通用数据与训练设置
DATA_DIR = 'data'
MAX_EPOCHS = 2400
INITIAL_LR = 0.0001
MIN_LR = 1e-5
LAST_K = 5
EVAL_RUNS = 30
STOCHASTIC_EVAL = True

# visdom
USE_VISDOM = True
VISDOM_ENV_PREFIX = 'ablation_suite'

# baseline 原始输出目录（旧逻辑保留）
BASELINE_RESULT_DIR = os.path.join('result', 'round_robin_24cases')

# ablation 输出总目录（旧逻辑保留）
ABLATION_ROOT_DIR = os.path.join('result', 'ablation_suite')

# ---------------------------------------------------------
# DSLFMAE 表13样式训练配置（保留）
# ---------------------------------------------------------
DSLFMAE_DATASET_SOURCE = 'data'
DSLFMAE_RESULT_DIR = os.path.join('result', 'dslfmae_table13_like')
DSLFMAE_MAX_EPOCHS = 2700
DSLFMAE_TAIL_K = 10

# ---------------------------------------------------------
# 新增：DSLFMAE 已训练模型评测配置
# ---------------------------------------------------------
DSLFMAE_MODEL_PATH = os.path.join(DSLFMAE_RESULT_DIR, 'best_model.pth')
DSLFMAE_EVAL_DIR = os.path.join(DSLFMAE_RESULT_DIR, 'saved_model_eval')
DSLFMAE_EVAL_RUNS = 30
DSLFMAE_EVAL_STOCHASTIC = True

# ---------------------------------------------------------
# ablation 方案定义（旧逻辑保留）
# ---------------------------------------------------------
ABLATION_VARIANTS = {
    'baseline': {
        'use_fluid_mask': True,
        'use_fluid_state': True,
        'use_fluid_reward_scale': True
    },
    'ablation_no_zero_mask': {
        'use_fluid_mask': False,
        'use_fluid_state': True,
        'use_fluid_reward_scale': True
    },
    'ablation_no_fluid_state': {
        'use_fluid_mask': True,
        'use_fluid_state': False,
        'use_fluid_reward_scale': True
    },
    'ablation_no_reward_scale': {
        'use_fluid_mask': True,
        'use_fluid_state': True,
        'use_fluid_reward_scale': False
    }
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_csv_rows(rows, save_path):
    folder = os.path.dirname(save_path)
    if folder:
        ensure_dir(folder)

    if not rows:
        return

    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj, save_path):
    folder = os.path.dirname(save_path)
    if folder:
        ensure_dir(folder)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def moving_average(data, window_size=20):
    if len(data) == 0:
        return np.array([])
    res = np.zeros(len(data), dtype=np.float64)
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        res[i] = np.mean(data[start:i + 1])
    return res


def export_schedule_detail(schedule_log,
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
            'compact_machine_label': machine_labels[compact_machine] if machine_labels is not None and compact_machine < len(machine_labels) else 'M{}'.format(compact_machine + 1),
            'type': int(item['type']),
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

        rows.append(row)

    save_csv_rows(rows, save_path)
    return save_path


def save_machine_mapping_csv(machine_labels,
                             original_machine_labels,
                             compact_to_original_machine_id,
                             save_dir='result',
                             filename='machine_mapping.csv'):
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, filename)

    rows = []
    for compact_id, old_m in enumerate(compact_to_original_machine_id):
        rows.append({
            'compact_machine_id': compact_id + 1,
            'compact_machine_label': machine_labels[compact_id] if compact_id < len(machine_labels) else 'M{}'.format(compact_id + 1),
            'original_machine_id': int(old_m) + 1,
            'original_machine_label': original_machine_labels[compact_id] if compact_id < len(original_machine_labels) else 'M{}'.format(int(old_m) + 1)
        })

    save_csv_rows(rows, save_path)
    return save_path


def save_gantt_chart(schedule_log,
                     num_machines,
                     makespan,
                     save_dir='result',
                     filename='best_gantt.png',
                     machine_labels=None,
                     drop_idle_machines=False):
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, filename)

    if len(schedule_log) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title('Empty Schedule')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    if drop_idle_machines:
        machine_list = sorted(list(set(int(e['machine']) for e in schedule_log)))
    else:
        machine_list = list(range(int(num_machines)))

    machine_pos = {}
    for idx, m in enumerate(machine_list):
        machine_pos[m] = idx

    fig, ax = plt.subplots(figsize=(14, 7))
    type_op_pairs = sorted(list(set((int(entry['type']), int(entry['op'])) for entry in schedule_log)))
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    color_map = {}
    for i, key in enumerate(type_op_pairs):
        color_map[key] = palette[i % len(palette)]

    label_count_per_machine = {}
    for m in machine_list:
        label_count_per_machine[m] = 0
    label_duration_threshold = max(50.0, makespan * 0.015)
    max_labels_per_machine = 16

    schedule_sorted = sorted(schedule_log, key=lambda x: (x['machine'], x['start'], x['end']))

    for entry in schedule_sorted:
        m = int(entry['machine'])
        if m not in machine_pos:
            continue

        r = int(entry['type'])
        j = int(entry['op'])
        start = float(entry['start'])
        end = float(entry['end'])
        duration = end - start
        y = machine_pos[m]

        ax.barh(
            y=y,
            width=duration,
            left=start,
            height=0.65,
            color=color_map[(r, j)],
            edgecolor='black',
            alpha=0.88
        )

        if duration >= label_duration_threshold and label_count_per_machine[m] < max_labels_per_machine:
            label = 'T{}-Op{}'.format(r, j)
            if 'job_idx_in_type' in entry:
                label = 'T{}-J{}-Op{}'.format(int(entry['type']), int(entry['job_idx_in_type']), int(entry['op']))
            ax.text(
                start + duration / 2.0,
                y,
                label,
                ha='center',
                va='center',
                color='white',
                fontsize=8,
                fontweight='bold'
            )
            label_count_per_machine[m] += 1

    ax.set_yticks(range(len(machine_list)))

    if machine_labels is not None:
        labels = []
        for m in machine_list:
            if 0 <= m < len(machine_labels):
                labels.append(machine_labels[m])
            else:
                labels.append('M{}'.format(m + 1))
        ax.set_yticklabels(labels, fontsize=11)
    else:
        ax.set_yticklabels(['M{}'.format(m + 1) for m in machine_list], fontsize=11)

    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Machine', fontsize=14)
    ax.set_title('Best Schedule Gantt Chart (Makespan: {:.2f})'.format(makespan), fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def save_instance_curve_png(instance_id, y_values, save_dir='result', filename=None):
    if filename is None:
        filename = '{}_curve.png'.format(instance_id)

    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, filename)

    if len(y_values) == 0:
        return None

    x_values = np.arange(1, len(y_values) + 1)
    ma_values = moving_average(y_values, window_size=min(5, len(y_values)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, y_values, linewidth=1.2, alpha=0.75, label='Current objective')
    ax.plot(x_values, ma_values, linewidth=2.0, label='Moving average')
    ax.set_xlabel('Occurrence index')
    ax.set_ylabel('Makespan')
    ax.set_title('{} training curve'.format(instance_id))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return save_path


def init_visdom(env_name):
    if Visdom is None:
        print('Visdom 未安装，跳过可视化。')
        return None

    try:
        vis = Visdom(env=env_name)
        ok = vis.check_connection(timeout_seconds=1)
        if not ok:
            print('Visdom 服务未连接，跳过可视化。')
            return None
        print('Visdom 已连接，env = {}'.format(env_name))
        return vis
    except Exception as e:
        print('Visdom 初始化失败: {}'.format(str(e)))
        return None


def update_visdom_curve(vis, win_map, instance_id, x, y):
    if vis is None:
        return

    win_key = 'curve_{}'.format(instance_id)
    title = '{} training curve'.format(instance_id)

    if win_key not in win_map:
        win_map[win_key] = vis.line(
            X=np.array([x], dtype=np.float32),
            Y=np.array([y], dtype=np.float32),
            opts=dict(
                title=title,
                xlabel='Occurrence index',
                ylabel='Makespan'
            )
        )
    else:
        vis.line(
            X=np.array([x], dtype=np.float32),
            Y=np.array([y], dtype=np.float32),
            win=win_map[win_key],
            update='append'
        )


def get_edge_dim_from_instance(data):
    """工业订单启用 changeover 弧特征时，edge_dim 从 6 扩展为 7。"""
    if bool(data.get('is_industrial_instance', False)) and bool(data.get('use_changeover_feature', False)):
        return 7
    return 6


def load_policy_state_dict_compatible(agent, state_dict):
    """
    正常情况下严格加载模型。
    当仅把 edge_dim 从 6 扩展到 7 时，部分 Linear 输入维度会增加 1。
    这里复制旧权重的前置列，新增 changeover 列保留当前初始化值，
    从而保证旧的 6 维模型仍可用于工业实例评测/热启动。
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

            # 兼容 edge_dim 6 -> 7：Linear weight 的输入列数增加。
            if (source_tensor.ndim == 2 and target_tensor.ndim == 2 and
                    source_tensor.shape[0] == target_tensor.shape[0] and
                    source_tensor.shape[1] < target_tensor.shape[1]):
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


def build_agent(initial_lr=0.0001, edge_dim=6):
    agent = PPO_Agent(
        op_dim=8,
        m_dim=7,
        edge_dim=int(edge_dim),
        lr=initial_lr,
        batch_size=256
    )
    return agent


def make_env(data, ablation_cfg):
    return AblationFJSPEnv(copy.deepcopy(data), ablation_cfg=ablation_cfg)


def run_policy_inference_once(agent, instance_data, ablation_cfg, stochastic=False):
    infer_env = make_env(instance_data, ablation_cfg)
    state = infer_env.reset()

    start_time = time.time()

    while True:
        if stochastic:
            idx, _, _ = agent.select_action(state)
        else:
            idx = agent.evaluate_action(state)

        if idx is None:
            break

        state, _, done, _ = infer_env.step(state['valid_actions'][idx])
        if done:
            break

    cpu_time = time.time() - start_time

    return {
        'makespan': float(infer_env.current_time),
        'cpu_time': float(cpu_time),
        'schedule_log': copy.deepcopy(infer_env.schedule_log),
        'num_machines': int(infer_env.num_machines),
        'machine_labels': infer_env.data.get('machine_labels', None),
        'original_machine_labels': infer_env.data.get('original_machine_labels', None),
        'compact_to_original_machine_id': infer_env.data.get('compact_to_original_machine_id', None)
    }


# =========================================================
# 通用 round-robin 训练逻辑
# =========================================================
def train_round_robin_on_instance_list(instance_list,
                                       max_epochs,
                                       result_dir,
                                       variant_name,
                                       ablation_cfg,
                                       visdom_env_name,
                                       curve_window_last_k=5):
    per_instance_dir = os.path.join(result_dir, 'per_instance')
    history_csv = os.path.join(result_dir, 'round_robin_epoch_log.csv')
    round_summary_csv = os.path.join(result_dir, 'round_summary.csv')
    instance_summary_csv = os.path.join(result_dir, 'instance_summary.csv')
    best_model_path = os.path.join(result_dir, 'best_model.pth')
    final_model_path = os.path.join(result_dir, 'final_model.pth')
    config_json = os.path.join(result_dir, 'ablation_config.json')

    ensure_dir(result_dir)
    ensure_dir(per_instance_dir)

    save_json({
        'variant_name': variant_name,
        'ablation_cfg': ablation_cfg,
        'max_epochs': max_epochs,
        'initial_lr': INITIAL_LR,
        'min_lr': MIN_LR,
        'eval_runs': EVAL_RUNS,
        'stochastic_eval': STOCHASTIC_EVAL
    }, config_json)

    num_instances = len(instance_list)
    if num_instances <= 0:
        raise ValueError('instance_list 为空')
    if max_epochs % num_instances != 0:
        raise ValueError('max_epochs 必须能被实例数整除: {} vs {}'.format(max_epochs, num_instances))

    expected_points_per_instance = max_epochs // num_instances

    print('\n==============================================================')
    print('开始训练: {}'.format(variant_name))
    print('result_dir = {}'.format(result_dir))
    print('ablation_cfg = {}'.format(ablation_cfg))
    print('实例数 = {}'.format(num_instances))
    print('总训练周期 = {}, 每个实例将获得 {} 个数据点'.format(max_epochs, expected_points_per_instance))
    print('==============================================================')

    train_edge_dim = get_edge_dim_from_instance(instance_list[0][1])
    agent = build_agent(initial_lr=INITIAL_LR, edge_dim=train_edge_dim)

    print('edge_dim = {}'.format(train_edge_dim))

    vis = init_visdom(visdom_env_name) if USE_VISDOM else None
    vis_windows = {}

    global_epoch_rows = []
    round_rows = []

    instance_stats = {}
    for instance_id, _ in instance_list:
        instance_folder = os.path.join(per_instance_dir, instance_id)
        ensure_dir(instance_folder)

        instance_stats[instance_id] = {
            'occurrence_count': 0,
            'values': [],
            'times': [],
            'epochs': [],
            'losses': [],
            'rows': [],
            'best_cmax': float('inf'),
            'best_epoch': None,
            'best_occurrence': None,
            'best_schedule_log': [],
            'best_schedule_csv_path': '',
            'best_gantt_path': '',
            'curve_png_path': ''
        }

    current_round_eval_values = []
    best_round_avg = float('inf')

    for epoch in range(1, max_epochs + 1):
        round_id = (epoch - 1) // num_instances + 1
        round_pos = (epoch - 1) % num_instances
        instance_id, train_data = instance_list[round_pos]
        stats = instance_stats[instance_id]

        stats['occurrence_count'] += 1
        occurrence_idx = stats['occurrence_count']

        frac = 1.0 - (epoch - 1.0) / max_epochs
        current_lr = max(INITIAL_LR * frac, MIN_LR)
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = current_lr

        cycle_start_time = time.time()

        train_env = make_env(train_data, ablation_cfg)
        state = train_env.reset()

        while True:
            idx, log_prob, val = agent.select_action(state)
            if idx is None:
                break

            next_state, reward, done, _ = train_env.step(state['valid_actions'][idx])
            agent.store_transition((state, idx, log_prob, reward, done, val))
            state = next_state

            if done:
                break

        train_rollout_cmax = float(train_env.current_time)
        loss = agent.update()
        train_time_sec = time.time() - cycle_start_time

        eval_res = run_policy_inference_once(agent, train_data, ablation_cfg, stochastic=False)
        eval_cmax = float(eval_res['makespan'])
        eval_time_sec = float(eval_res['cpu_time'])
        cycle_time_sec = float(train_time_sec + eval_time_sec)

        stats['values'].append(eval_cmax)
        stats['times'].append(cycle_time_sec)
        stats['epochs'].append(epoch)
        stats['losses'].append(loss)

        running_cb = float(np.min(stats['values']))
        running_ca = float(np.mean(stats['values']))

        row = {
            'variant_name': variant_name,
            'epoch': int(epoch),
            'round_id': int(round_id),
            'round_pos': int(round_pos + 1),
            'instance_id': instance_id,
            'instance_occurrence': int(occurrence_idx),
            'lr': float(current_lr),
            'train_loss': float(loss),
            'train_rollout_cmax': float(train_rollout_cmax),
            'current_objective': float(eval_cmax),
            'train_time_sec': float(train_time_sec),
            'eval_time_sec': float(eval_time_sec),
            'cycle_time_sec': float(cycle_time_sec),
            'running_cb': float(running_cb),
            'running_ca': float(running_ca)
        }

        if 'dslfmae_M' in train_data:
            row['M'] = int(train_data['dslfmae_M'])
            row['R'] = int(train_data['dslfmae_R'])
            row['N'] = int(train_data['dslfmae_N'])

        stats['rows'].append(row)
        global_epoch_rows.append(row)
        current_round_eval_values.append(eval_cmax)

        if eval_cmax < stats['best_cmax']:
            stats['best_cmax'] = eval_cmax
            stats['best_epoch'] = epoch
            stats['best_occurrence'] = occurrence_idx
            stats['best_schedule_log'] = copy.deepcopy(eval_res['schedule_log'])

            instance_folder = os.path.join(per_instance_dir, instance_id)
            best_schedule_csv = export_schedule_detail(
                stats['best_schedule_log'],
                save_dir=instance_folder,
                filename='best_schedule_detail.csv',
                machine_labels=eval_res['machine_labels'],
                original_machine_labels=eval_res['original_machine_labels'],
                compact_to_original_machine_id=eval_res['compact_to_original_machine_id']
            )
            best_gantt_png = save_gantt_chart(
                stats['best_schedule_log'],
                eval_res['num_machines'],
                eval_cmax,
                save_dir=instance_folder,
                filename='best_gantt.png',
                machine_labels=eval_res['machine_labels'],
                drop_idle_machines=False
            )

            stats['best_schedule_csv_path'] = best_schedule_csv
            stats['best_gantt_path'] = best_gantt_png

        save_csv_rows(global_epoch_rows, history_csv)

        instance_history_csv = os.path.join(per_instance_dir, instance_id, 'history.csv')
        save_csv_rows(stats['rows'], instance_history_csv)

        update_visdom_curve(
            vis=vis,
            win_map=vis_windows,
            instance_id=instance_id,
            x=occurrence_idx,
            y=eval_cmax
        )

        if (epoch % num_instances) == 0:
            round_avg = float(np.mean(current_round_eval_values))
            round_cb = float(np.min(current_round_eval_values))

            round_row = {
                'variant_name': variant_name,
                'round_id': int(round_id),
                'start_epoch': int(epoch - num_instances + 1),
                'end_epoch': int(epoch),
                'round_avg_objective': float(round_avg),
                'round_cb_objective': float(round_cb)
            }
            round_rows.append(round_row)
            save_csv_rows(round_rows, round_summary_csv)

            if round_avg < best_round_avg:
                best_round_avg = round_avg
                torch.save(agent.policy.state_dict(), best_model_path)

            print(
                '[{}] Round {:3d}/{} | Epoch {:4d}/{:4d} | Round Avg: {:10.2f} | Round Cb: {:10.2f} | Best Round Avg: {:10.2f}'.format(
                    variant_name,
                    round_id,
                    max_epochs // num_instances,
                    epoch,
                    max_epochs,
                    round_avg,
                    round_cb,
                    best_round_avg
                )
            )
            current_round_eval_values = []
        else:
            print(
                '[{}] Epoch {:4d}/{:4d} | Case {:>12s} | Point {:3d}/{:3d} | Obj {:10.2f} | Cb {:10.2f} | Ca {:10.2f} | Time {:7.3f}s'.format(
                    variant_name,
                    epoch,
                    max_epochs,
                    instance_id,
                    occurrence_idx,
                    expected_points_per_instance,
                    eval_cmax,
                    running_cb,
                    running_ca,
                    cycle_time_sec
                )
            )

    torch.save(agent.policy.state_dict(), final_model_path)

    summary_rows = []
    for instance_id, data in instance_list:
        stats = instance_stats[instance_id]
        values = stats['values']
        times = stats['times']

        if len(values) == 0:
            continue

        tail_values = values[-curve_window_last_k:] if len(values) >= curve_window_last_k else values[:]

        curve_png = save_instance_curve_png(
            instance_id=instance_id,
            y_values=values,
            save_dir=os.path.join(per_instance_dir, instance_id),
            filename='curve.png'
        )
        stats['curve_png_path'] = curve_png if curve_png is not None else ''

        row = {
            'variant_name': variant_name,
            'instance_id': instance_id,
            'points': int(len(values)),
            'best_epoch': int(stats['best_epoch']) if stats['best_epoch'] is not None else '',
            'best_occurrence': int(stats['best_occurrence']) if stats['best_occurrence'] is not None else '',
            'cb_all': float(np.min(values)),
            'ca_all': float(np.mean(values)),
            'cb_last_k': float(np.min(tail_values)),
            'ca_last_k': float(np.mean(tail_values)),
            'avg_cycle_time_sec': float(np.mean(times)),
            'best_schedule_csv': stats['best_schedule_csv_path'],
            'best_gantt_png': stats['best_gantt_path'],
            'curve_png': stats['curve_png_path']
        }

        if 'dslfmae_M' in data:
            row['M'] = int(data['dslfmae_M'])
            row['R'] = int(data['dslfmae_R'])
            row['N'] = int(data['dslfmae_N'])

        summary_rows.append(row)

    save_csv_rows(summary_rows, instance_summary_csv)

    print('\n✅ 训练完成: {}'.format(variant_name))
    print('result_dir: {}'.format(result_dir))
    print('best_model: {}'.format(best_model_path))
    print('final_model: {}'.format(final_model_path))


# =========================================================
# 旧逻辑：24 标准算例训练
# =========================================================
def train_round_robin_24cases():
    standard_instances = load_all_standard_instances(DATA_DIR)
    train_round_robin_on_instance_list(
        instance_list=standard_instances,
        max_epochs=MAX_EPOCHS,
        result_dir=BASELINE_RESULT_DIR,
        variant_name='baseline',
        ablation_cfg=ABLATION_VARIANTS['baseline'],
        visdom_env_name='round_robin_24cases',
        curve_window_last_k=LAST_K
    )


def evaluate_saved_model_on_public_instances():
    result_dir = BASELINE_RESULT_DIR
    variant_name = 'baseline'
    ablation_cfg = ABLATION_VARIANTS['baseline']

    model_path = os.path.join(result_dir, 'best_model.pth')
    eval_result_dir = os.path.join(result_dir, 'benchmark_eval')
    eval_per_instance_dir = os.path.join(eval_result_dir, 'per_instance')
    eval_summary_csv = os.path.join(eval_result_dir, 'public_benchmark_summary.csv')
    eval_all_runs_csv = os.path.join(eval_result_dir, 'public_benchmark_all_runs.csv')

    ensure_dir(eval_result_dir)
    ensure_dir(eval_per_instance_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError('未找到 best_model.pth: {}'.format(model_path))

    standard_instances = load_all_standard_instances(DATA_DIR)
    agent = build_agent(initial_lr=INITIAL_LR)

    state_dict = torch.load(model_path, map_location=agent.device)
    load_policy_state_dict_compatible(agent, state_dict)
    agent.policy.eval()

    all_run_rows = []
    summary_rows = []

    print('\n🚀 开始评测 public benchmark: {}'.format(variant_name))

    for instance_id, instance_data in standard_instances:
        instance_folder = os.path.join(eval_per_instance_dir, instance_id)
        ensure_dir(instance_folder)

        cmax_list = []
        tcpu_list = []
        best_cmax = float('inf')
        best_res = None
        instance_run_rows = []

        for run_id in range(1, EVAL_RUNS + 1):
            res = run_policy_inference_once(
                agent=agent,
                instance_data=instance_data,
                ablation_cfg=ablation_cfg,
                stochastic=STOCHASTIC_EVAL
            )

            cmax = float(res['makespan'])
            tcpu = float(res['cpu_time'])

            cmax_list.append(cmax)
            tcpu_list.append(tcpu)

            run_row = {
                'variant_name': variant_name,
                'instance_id': instance_id,
                'run_id': int(run_id),
                'eval_mode': 'stochastic' if STOCHASTIC_EVAL else 'greedy',
                'makespan': float(cmax),
                'cpu_time_sec': float(tcpu)
            }
            all_run_rows.append(run_row)
            instance_run_rows.append(run_row)

            if cmax < best_cmax:
                best_cmax = cmax
                best_res = copy.deepcopy(res)

        cb = float(np.min(cmax_list))
        ca = float(np.mean(cmax_list))
        tcpu_avg = float(np.mean(tcpu_list))

        best_schedule_csv = export_schedule_detail(
            best_res['schedule_log'],
            save_dir=instance_folder,
            filename='best_schedule_detail.csv',
            machine_labels=best_res['machine_labels'],
            original_machine_labels=best_res['original_machine_labels'],
            compact_to_original_machine_id=best_res['compact_to_original_machine_id']
        )
        best_gantt_png = save_gantt_chart(
            best_res['schedule_log'],
            best_res['num_machines'],
            best_res['makespan'],
            save_dir=instance_folder,
            filename='best_gantt.png',
            machine_labels=best_res['machine_labels'],
            drop_idle_machines=False
        )

        instance_runs_csv = os.path.join(instance_folder, 'all_runs.csv')
        save_csv_rows(instance_run_rows, instance_runs_csv)

        summary_rows.append({
            'variant_name': variant_name,
            'instance_id': instance_id,
            'runs': int(EVAL_RUNS),
            'eval_mode': 'stochastic' if STOCHASTIC_EVAL else 'greedy',
            'cb': float(cb),
            'ca': float(ca),
            'tcpu_avg_sec': float(tcpu_avg),
            'best_schedule_csv': best_schedule_csv,
            'best_gantt_png': best_gantt_png
        })

        print('[{}][SUMMARY] {} | Cb {:10.2f} | Ca {:10.2f} | Tcpu {:8.4f}s'.format(
            variant_name, instance_id, cb, ca, tcpu_avg
        ))

    save_csv_rows(all_run_rows, eval_all_runs_csv)
    save_csv_rows(summary_rows, eval_summary_csv)

    print('\n✅ public benchmark 评测完成: {}'.format(variant_name))
    print('全部运行明细: {}'.format(eval_all_runs_csv))
    print('实例汇总: {}'.format(eval_summary_csv))


def evaluate_saved_model_on_industrial_orders():
    industrial_dir = os.path.join('data', 'instance')
    result_dir = BASELINE_RESULT_DIR
    variant_name = 'baseline'
    ablation_cfg = ABLATION_VARIANTS['baseline']
    model_path = os.path.join(result_dir, 'best_model.pth')

    out_dir = os.path.join(result_dir, 'industrial_eval')
    per_order_dir = os.path.join(out_dir, 'per_order')
    summary_csv = os.path.join(out_dir, 'industrial_table16_style_summary.csv')
    all_runs_csv = os.path.join(out_dir, 'industrial_all_runs.csv')

    runs = 30
    stochastic_eval = True

    ensure_dir(out_dir)
    ensure_dir(per_order_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError('未找到 best_model.pth: {}'.format(model_path))

    order_instances = load_industrial_orders(
        instance_dir=industrial_dir,
        compact_unused_machines=True,
        drop_zero_job_types=True
    )

    industrial_edge_dim = get_edge_dim_from_instance(order_instances[0][1]) if len(order_instances) > 0 else 6
    agent = build_agent(initial_lr=INITIAL_LR, edge_dim=industrial_edge_dim)
    state_dict = torch.load(model_path, map_location=agent.device)
    load_policy_state_dict_compatible(agent, state_dict)
    agent.policy.eval()

    all_run_rows = []
    summary_rows = []

    print('\n🚀 开始工业订单评测: {}'.format(variant_name))

    for order_name, order_data in order_instances:
        order_folder = os.path.join(per_order_dir, order_name)
        ensure_dir(order_folder)

        counts = order_data.get('order_kind_counts', ['', '', ''])

        print('\n{}:'.format(order_name))
        print('  kind_counts             = {}'.format(counts))
        print('  active_job_types        = {}'.format(order_data['num_job_types']))
        print('  active_machines         = {}'.format(order_data['num_machines']))
        print('  compact_machine_labels  = {}'.format(order_data.get('machine_labels', [])))
        print('  original_machine_labels = {}'.format(order_data.get('original_machine_labels', [])))
        print('  edge_dim                = {}'.format(get_edge_dim_from_instance(order_data)))
        print('  changeover_feature      = {}'.format(bool(order_data.get('use_changeover_feature', False))))
        print('  has_changeover_data     = {}'.format(bool(order_data.get('has_changeover_data', False))))

        machine_mapping_csv = save_machine_mapping_csv(
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

        for run_id in range(1, runs + 1):
            res = run_policy_inference_once(
                agent=agent,
                instance_data=order_data,
                ablation_cfg=ablation_cfg,
                stochastic=stochastic_eval
            )

            cmax = float(res['makespan'])
            tcpu = float(res['cpu_time'])

            cmax_list.append(cmax)
            tcpu_list.append(tcpu)

            row = {
                'variant_name': variant_name,
                'order': order_name,
                'run_id': int(run_id),
                'eval_mode': 'stochastic',
                'makespan': cmax,
                'cpu_time_sec': tcpu
            }
            order_run_rows.append(row)
            all_run_rows.append(row)

            if cmax < best_cmax:
                best_cmax = cmax
                best_res = copy.deepcopy(res)

            print('  run {:02d}/{} | Cmax = {:.2f} | Tcpu = {:.4f}s'.format(
                run_id, runs, cmax, tcpu
            ))

        cb = float(np.min(cmax_list))
        ca = float(np.mean(cmax_list))
        tcpu_avg = float(np.mean(tcpu_list))

        order_runs_csv = os.path.join(order_folder, 'all_runs.csv')
        save_csv_rows(order_run_rows, order_runs_csv)

        best_schedule_csv = export_schedule_detail(
            best_res['schedule_log'],
            save_dir=order_folder,
            filename='best_schedule_detail.csv',
            machine_labels=best_res['machine_labels'],
            original_machine_labels=best_res['original_machine_labels'],
            compact_to_original_machine_id=best_res['compact_to_original_machine_id']
        )

        best_gantt_full_active = save_gantt_chart(
            best_res['schedule_log'],
            best_res['num_machines'],
            best_res['makespan'],
            save_dir=order_folder,
            filename='best_gantt_full_active.png',
            machine_labels=best_res['machine_labels'],
            drop_idle_machines=False
        )

        best_gantt_used_only = save_gantt_chart(
            best_res['schedule_log'],
            best_res['num_machines'],
            best_res['makespan'],
            save_dir=order_folder,
            filename='best_gantt_used_only.png',
            machine_labels=best_res['machine_labels'],
            drop_idle_machines=True
        )

        summary_rows.append({
            'variant_name': variant_name,
            'order': order_name,
            'kind0_count': counts[0] if len(counts) > 0 else '',
            'kind1_count': counts[1] if len(counts) > 1 else '',
            'kind2_count': counts[2] if len(counts) > 2 else '',
            'runs': int(runs),
            'Cb': cb,
            'Ca': ca,
            'Tcpu_avg_sec': tcpu_avg,
            'best_schedule_csv': best_schedule_csv,
            'machine_mapping_csv': machine_mapping_csv,
            'best_gantt_full_active': best_gantt_full_active,
            'best_gantt_used_only': best_gantt_used_only
        })

        print('  [SUMMARY] Cb = {:.2f}, Ca = {:.2f}, Tcpu = {:.4f}s'.format(
            cb, ca, tcpu_avg
        ))

    save_csv_rows(all_run_rows, all_runs_csv)
    save_csv_rows(summary_rows, summary_csv)

    print('\n✅ 工业订单评测完成: {}'.format(variant_name))
    print('全部运行明细: {}'.format(all_runs_csv))
    print('表16风格汇总: {}'.format(summary_csv))


# =========================================================
# 消融逻辑（保留）
# =========================================================
def train_single_variant():
    cfg = ABLATION_VARIANTS[SINGLE_VARIANT_NAME]
    result_dir = os.path.join(ABLATION_ROOT_DIR, SINGLE_VARIANT_NAME)
    standard_instances = load_all_standard_instances(DATA_DIR)

    train_round_robin_on_instance_list(
        instance_list=standard_instances,
        max_epochs=MAX_EPOCHS,
        result_dir=result_dir,
        variant_name=SINGLE_VARIANT_NAME,
        ablation_cfg=cfg,
        visdom_env_name='{}__{}'.format(VISDOM_ENV_PREFIX, SINGLE_VARIANT_NAME),
        curve_window_last_k=LAST_K
    )


def eval_single_variant():
    cfg = ABLATION_VARIANTS[SINGLE_VARIANT_NAME]
    result_dir = os.path.join(ABLATION_ROOT_DIR, SINGLE_VARIANT_NAME)
    model_path = os.path.join(result_dir, 'best_model.pth')

    eval_result_dir = os.path.join(result_dir, 'benchmark_eval')
    eval_per_instance_dir = os.path.join(eval_result_dir, 'per_instance')
    eval_summary_csv = os.path.join(eval_result_dir, 'public_benchmark_summary.csv')
    eval_all_runs_csv = os.path.join(eval_result_dir, 'public_benchmark_all_runs.csv')

    ensure_dir(eval_result_dir)
    ensure_dir(eval_per_instance_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError('未找到 best_model.pth: {}'.format(model_path))

    standard_instances = load_all_standard_instances(DATA_DIR)
    agent = build_agent(initial_lr=INITIAL_LR)

    state_dict = torch.load(model_path, map_location=agent.device)
    load_policy_state_dict_compatible(agent, state_dict)
    agent.policy.eval()

    all_run_rows = []
    summary_rows = []

    print('\n🚀 开始评测 public benchmark: {}'.format(SINGLE_VARIANT_NAME))

    for instance_id, instance_data in standard_instances:
        instance_folder = os.path.join(eval_per_instance_dir, instance_id)
        ensure_dir(instance_folder)

        cmax_list = []
        tcpu_list = []
        best_cmax = float('inf')
        best_res = None
        instance_run_rows = []

        for run_id in range(1, EVAL_RUNS + 1):
            res = run_policy_inference_once(
                agent=agent,
                instance_data=instance_data,
                ablation_cfg=cfg,
                stochastic=STOCHASTIC_EVAL
            )

            cmax = float(res['makespan'])
            tcpu = float(res['cpu_time'])

            cmax_list.append(cmax)
            tcpu_list.append(tcpu)

            run_row = {
                'variant_name': SINGLE_VARIANT_NAME,
                'instance_id': instance_id,
                'run_id': int(run_id),
                'eval_mode': 'stochastic' if STOCHASTIC_EVAL else 'greedy',
                'makespan': float(cmax),
                'cpu_time_sec': float(tcpu)
            }
            all_run_rows.append(run_row)
            instance_run_rows.append(run_row)

            if cmax < best_cmax:
                best_cmax = cmax
                best_res = copy.deepcopy(res)

        cb = float(np.min(cmax_list))
        ca = float(np.mean(cmax_list))
        tcpu_avg = float(np.mean(tcpu_list))

        best_schedule_csv = export_schedule_detail(
            best_res['schedule_log'],
            save_dir=instance_folder,
            filename='best_schedule_detail.csv'
        )
        best_gantt_png = save_gantt_chart(
            best_res['schedule_log'],
            best_res['num_machines'],
            best_res['makespan'],
            save_dir=instance_folder,
            filename='best_gantt.png'
        )

        instance_runs_csv = os.path.join(instance_folder, 'all_runs.csv')
        save_csv_rows(instance_run_rows, instance_runs_csv)

        summary_rows.append({
            'variant_name': SINGLE_VARIANT_NAME,
            'instance_id': instance_id,
            'runs': int(EVAL_RUNS),
            'eval_mode': 'stochastic' if STOCHASTIC_EVAL else 'greedy',
            'cb': float(cb),
            'ca': float(ca),
            'tcpu_avg_sec': float(tcpu_avg),
            'best_schedule_csv': best_schedule_csv,
            'best_gantt_png': best_gantt_png
        })

        print('[{}][SUMMARY] {} | Cb {:10.2f} | Ca {:10.2f} | Tcpu {:8.4f}s'.format(
            SINGLE_VARIANT_NAME, instance_id, cb, ca, tcpu_avg
        ))

    save_csv_rows(all_run_rows, eval_all_runs_csv)
    save_csv_rows(summary_rows, eval_summary_csv)

    print('\n✅ public benchmark 评测完成: {}'.format(SINGLE_VARIANT_NAME))
    print('全部运行明细: {}'.format(eval_all_runs_csv))
    print('实例汇总: {}'.format(eval_summary_csv))


def train_and_eval_single():
    train_single_variant()
    eval_single_variant()


def run_public_ablation_suite():
    ensure_dir(ABLATION_ROOT_DIR)
    suite_summary_rows = []

    for variant_name, ablation_cfg in ABLATION_VARIANTS.items():
        result_dir = os.path.join(ABLATION_ROOT_DIR, variant_name)
        standard_instances = load_all_standard_instances(DATA_DIR)

        train_round_robin_on_instance_list(
            instance_list=standard_instances,
            max_epochs=MAX_EPOCHS,
            result_dir=result_dir,
            variant_name=variant_name,
            ablation_cfg=ablation_cfg,
            visdom_env_name='{}__{}'.format(VISDOM_ENV_PREFIX, variant_name),
            curve_window_last_k=LAST_K
        )

        model_path = os.path.join(result_dir, 'best_model.pth')
        eval_result_dir = os.path.join(result_dir, 'benchmark_eval')
        eval_per_instance_dir = os.path.join(eval_result_dir, 'per_instance')
        eval_summary_csv = os.path.join(eval_result_dir, 'public_benchmark_summary.csv')
        eval_all_runs_csv = os.path.join(eval_result_dir, 'public_benchmark_all_runs.csv')

        ensure_dir(eval_result_dir)
        ensure_dir(eval_per_instance_dir)

        agent = build_agent(initial_lr=INITIAL_LR)
        state_dict = torch.load(model_path, map_location=agent.device)
        load_policy_state_dict_compatible(agent, state_dict)
        agent.policy.eval()

        all_run_rows = []
        summary_rows = []

        print('\n🚀 开始评测 public benchmark: {}'.format(variant_name))

        for instance_id, instance_data in standard_instances:
            instance_folder = os.path.join(eval_per_instance_dir, instance_id)
            ensure_dir(instance_folder)

            cmax_list = []
            tcpu_list = []
            best_cmax = float('inf')
            best_res = None
            instance_run_rows = []

            for run_id in range(1, EVAL_RUNS + 1):
                res = run_policy_inference_once(
                    agent=agent,
                    instance_data=instance_data,
                    ablation_cfg=ablation_cfg,
                    stochastic=STOCHASTIC_EVAL
                )

                cmax = float(res['makespan'])
                tcpu = float(res['cpu_time'])

                cmax_list.append(cmax)
                tcpu_list.append(tcpu)

                run_row = {
                    'variant_name': variant_name,
                    'instance_id': instance_id,
                    'run_id': int(run_id),
                    'eval_mode': 'stochastic' if STOCHASTIC_EVAL else 'greedy',
                    'makespan': float(cmax),
                    'cpu_time_sec': float(tcpu)
                }
                all_run_rows.append(run_row)
                instance_run_rows.append(run_row)

                if cmax < best_cmax:
                    best_cmax = cmax
                    best_res = copy.deepcopy(res)

            cb = float(np.min(cmax_list))
            ca = float(np.mean(cmax_list))
            tcpu_avg = float(np.mean(tcpu_list))

            best_schedule_csv = export_schedule_detail(
                best_res['schedule_log'],
                save_dir=instance_folder,
                filename='best_schedule_detail.csv'
            )
            best_gantt_png = save_gantt_chart(
                best_res['schedule_log'],
                best_res['num_machines'],
                best_res['makespan'],
                save_dir=instance_folder,
                filename='best_gantt.png'
            )

            instance_runs_csv = os.path.join(instance_folder, 'all_runs.csv')
            save_csv_rows(instance_run_rows, instance_runs_csv)

            summary_rows.append({
                'variant_name': variant_name,
                'instance_id': instance_id,
                'runs': int(EVAL_RUNS),
                'eval_mode': 'stochastic' if STOCHASTIC_EVAL else 'greedy',
                'cb': float(cb),
                'ca': float(ca),
                'tcpu_avg_sec': float(tcpu_avg),
                'best_schedule_csv': best_schedule_csv,
                'best_gantt_png': best_gantt_png
            })

            print('[{}][SUMMARY] {} | Cb {:10.2f} | Ca {:10.2f} | Tcpu {:8.4f}s'.format(
                variant_name, instance_id, cb, ca, tcpu_avg
            ))

        save_csv_rows(all_run_rows, eval_all_runs_csv)
        save_csv_rows(summary_rows, eval_summary_csv)

        if os.path.exists(eval_summary_csv):
            with open(eval_summary_csv, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for row in rows:
                    suite_summary_rows.append(row)

    suite_summary_csv = os.path.join(ABLATION_ROOT_DIR, 'suite_public_benchmark_summary.csv')
    if len(suite_summary_rows) > 0:
        save_csv_rows(suite_summary_rows, suite_summary_csv)

    print('\n==============================================')
    print('✅ 全部 ablation 变体运行完成')
    print('总结果目录: {}'.format(ABLATION_ROOT_DIR))
    print('汇总对比表: {}'.format(suite_summary_csv))
    print('==============================================')


# =========================================================
# DSLFMAE 表13样式训练逻辑（保留）
# =========================================================
def _read_csv_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _parse_int(v, default=None):
    try:
        return int(float(v))
    except Exception:
        return default


def _parse_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def summarize_dslfmae_tail_statistics(result_dir, tail_k=10):
    per_instance_dir = os.path.join(result_dir, 'per_instance')
    summary_csv = os.path.join(result_dir, 'dslfmae_table13_like_summary.csv')
    instance_summary_csv = os.path.join(result_dir, 'dslfmae_instance_summary.csv')

    rows = []
    for instance_id in sorted(os.listdir(per_instance_dir)):
        hist_csv = os.path.join(per_instance_dir, instance_id, 'history.csv')
        if not os.path.exists(hist_csv):
            continue

        hist_rows = _read_csv_rows(hist_csv)
        if len(hist_rows) == 0:
            continue

        tail_rows = hist_rows[-tail_k:] if len(hist_rows) >= tail_k else hist_rows[:]

        current_values = []
        cycle_times = []
        M = None
        R = None
        N = None

        for r in tail_rows:
            obj = _parse_float(r.get('current_objective'), None)
            t = _parse_float(r.get('cycle_time_sec'), None)
            if obj is not None:
                current_values.append(obj)
            if t is not None:
                cycle_times.append(t)

            if M is None:
                M = _parse_int(r.get('M'), None)
            if R is None:
                R = _parse_int(r.get('R'), None)
            if N is None:
                N = _parse_int(r.get('N'), None)

        if len(current_values) == 0:
            continue

        mean_v = float(np.mean(current_values))
        std_v = float(np.std(current_values, ddof=0))
        cb_v = float(np.min(current_values))
        ca_v = float(np.mean(current_values))
        avg_time_v = float(np.mean(cycle_times)) if len(cycle_times) > 0 else 0.0

        rows.append({
            'instance_id': instance_id,
            'M': int(M) if M is not None else '',
            'R': int(R) if R is not None else '',
            'N': int(N) if N is not None else '',
            'tail_k': int(tail_k),
            'mean_last_k': mean_v,
            'std_last_k': std_v,
            'Cb_last_k': cb_v,
            'Ca_last_k': ca_v,
            'avg_cycle_time_last_k_sec': avg_time_v,
            'history_csv': hist_csv
        })

    rows = sorted(rows, key=lambda x: (int(x['M']), int(x['R']), int(x['N'])))
    save_csv_rows(rows, summary_csv)

    compact_rows = []
    for r in rows:
        compact_rows.append({
            'instance_id': r['instance_id'],
            'M': r['M'],
            'R': r['R'],
            'N': r['N'],
            'mean': r['mean_last_k'],
            'std': r['std_last_k'],
            'avg_cycle_time_sec': r['avg_cycle_time_last_k_sec']
        })
    save_csv_rows(compact_rows, instance_summary_csv)

    return summary_csv, instance_summary_csv


def plot_dslfmae_avg_computation_time(summary_csv, output_png):
    rows = _read_csv_rows(summary_csv)
    if len(rows) == 0:
        return None

    data = []
    for r in rows:
        item = {
            'instance_id': r['instance_id'],
            'M': _parse_int(r['M'], 0),
            'R': _parse_int(r['R'], 0),
            'N': _parse_int(r['N'], 0),
            'avg_time': _parse_float(r['avg_cycle_time_last_k_sec'], 0.0)
        }
        data.append(item)

    data = sorted(data, key=lambda x: (x['M'], x['R'], x['N']))

    Ms = sorted(list(set([d['M'] for d in data])))
    Rs = sorted(list(set([d['R'] for d in data])))
    Ns = sorted(list(set([d['N'] for d in data])))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for M in Ms:
        for N in Ns:
            sub = [d for d in data if d['M'] == M and d['N'] == N]
            sub = sorted(sub, key=lambda x: x['R'])
            if len(sub) <= 0:
                continue
            x = [d['R'] for d in sub]
            y = [d['avg_time'] for d in sub]
            ax.plot(x, y, marker='o', linewidth=1.5, label='M={}, N={}'.format(M, N))
    ax.set_title('R vs. Average Computation Time')
    ax.set_xlabel('R')
    ax.set_ylabel('Average Computation Time (s)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=8, ncol=1)

    ax = axes[1]
    for R in Rs:
        for N in Ns:
            sub = [d for d in data if d['R'] == R and d['N'] == N]
            sub = sorted(sub, key=lambda x: x['M'])
            if len(sub) <= 0:
                continue
            x = [d['M'] for d in sub]
            y = [d['avg_time'] for d in sub]
            ax.plot(x, y, marker='s', linewidth=1.5, label='R={}, N={}'.format(R, N))
    ax.set_title('M vs. Average Computation Time')
    ax.set_xlabel('M')
    ax.set_ylabel('Average Computation Time (s)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=8, ncol=1)

    ax = axes[2]
    for R in Rs:
        for M in Ms:
            sub = [d for d in data if d['R'] == R and d['M'] == M]
            sub = sorted(sub, key=lambda x: x['N'])
            if len(sub) <= 0:
                continue
            x = [d['N'] for d in sub]
            y = [d['avg_time'] for d in sub]
            ax.plot(x, y, marker='<', linewidth=1.5, label='R={}, M={}'.format(R, M))
    ax.set_title('N vs. Average Computation Time')
    ax.set_xlabel('N')
    ax.set_ylabel('Average Computation Time (s)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_png


def run_dslfmae_table13():
    instances = load_dslfmae_instances(
        dataset_source=DSLFMAE_DATASET_SOURCE
    )

    num_instances = len(instances)
    if num_instances <= 0:
        raise ValueError('DSLFMAE 实例列表为空')
    if DSLFMAE_MAX_EPOCHS % num_instances != 0:
        raise ValueError('DSLFMAE_MAX_EPOCHS 必须能被实例数整除: {} vs {}'.format(
            DSLFMAE_MAX_EPOCHS, num_instances
        ))

    print('\n==============================================================')
    print('开始运行 DSLFMAE 表13样式实验')
    print('dataset_source = {}'.format(DSLFMAE_DATASET_SOURCE))
    print('instances      = {}'.format(num_instances))
    print('max_epochs     = {}'.format(DSLFMAE_MAX_EPOCHS))
    print('tail_k         = {}'.format(DSLFMAE_TAIL_K))
    print('==============================================================')

    train_round_robin_on_instance_list(
        instance_list=instances,
        max_epochs=DSLFMAE_MAX_EPOCHS,
        result_dir=DSLFMAE_RESULT_DIR,
        variant_name='dslfmae_like_baseline',
        ablation_cfg=ABLATION_VARIANTS['baseline'],
        visdom_env_name='dslfmae_table13_like',
        curve_window_last_k=DSLFMAE_TAIL_K
    )

    summary_csv, compact_csv = summarize_dslfmae_tail_statistics(
        result_dir=DSLFMAE_RESULT_DIR,
        tail_k=DSLFMAE_TAIL_K
    )

    plot_png = plot_dslfmae_avg_computation_time(
        summary_csv=summary_csv,
        output_png=os.path.join(DSLFMAE_RESULT_DIR, 'dslfmae_avg_computation_time.png')
    )

    print('\n✅ DSLFMAE 表13样式实验完成')
    print('训练总日志: {}'.format(os.path.join(DSLFMAE_RESULT_DIR, 'round_robin_epoch_log.csv')))
    print('轮次汇总  : {}'.format(os.path.join(DSLFMAE_RESULT_DIR, 'round_summary.csv')))
    print('实例汇总  : {}'.format(os.path.join(DSLFMAE_RESULT_DIR, 'instance_summary.csv')))
    print('表格汇总  : {}'.format(summary_csv))
    print('紧凑表格  : {}'.format(compact_csv))
    print('时间图    : {}'.format(plot_png))


# =========================================================
# 新增：把 DSLFMAE 当前数据直接丢给已训练好的模型评测
# 按 P1-P24 的格式保存：
#   1) 每次运行结果
#   2) 每个实例 all_runs.csv
#   3) 最佳甘特图
#   4) 最佳调度明细
#   5) 汇总表格 CSV
#   6) 平均计算时间三联图
#   7) 表格 PNG
# =========================================================
def plot_dslfmae_avg_computation_time_from_eval_summary(summary_csv, output_png):
    rows = _read_csv_rows(summary_csv)
    if len(rows) == 0:
        return None

    data = []
    for r in rows:
        item = {
            'instance_id': r['instance_id'],
            'M': _parse_int(r['M'], 0),
            'R': _parse_int(r['R'], 0),
            'N': _parse_int(r['N'], 0),
            'avg_time': _parse_float(r['avg_cpu_time_sec'], 0.0)
        }
        data.append(item)

    data = sorted(data, key=lambda x: (x['M'], x['R'], x['N']))

    Ms = sorted(list(set([d['M'] for d in data])))
    Rs = sorted(list(set([d['R'] for d in data])))
    Ns = sorted(list(set([d['N'] for d in data])))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for M in Ms:
        for N in Ns:
            sub = [d for d in data if d['M'] == M and d['N'] == N]
            sub = sorted(sub, key=lambda x: x['R'])
            if len(sub) <= 0:
                continue
            x = [d['R'] for d in sub]
            y = [d['avg_time'] for d in sub]
            ax.plot(x, y, marker='o', linewidth=1.5, label='M={}, N={}'.format(M, N))
    ax.set_title('R vs. Average Computation Time')
    ax.set_xlabel('R')
    ax.set_ylabel('Average Computation Time (s)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=8, ncol=1)

    ax = axes[1]
    for R in Rs:
        for N in Ns:
            sub = [d for d in data if d['R'] == R and d['N'] == N]
            sub = sorted(sub, key=lambda x: x['M'])
            if len(sub) <= 0:
                continue
            x = [d['M'] for d in sub]
            y = [d['avg_time'] for d in sub]
            ax.plot(x, y, marker='s', linewidth=1.5, label='R={}, N={}'.format(R, N))
    ax.set_title('M vs. Average Computation Time')
    ax.set_xlabel('M')
    ax.set_ylabel('Average Computation Time (s)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=8, ncol=1)

    ax = axes[2]
    for R in Rs:
        for M in Ms:
            sub = [d for d in data if d['R'] == R and d['M'] == M]
            sub = sorted(sub, key=lambda x: x['N'])
            if len(sub) <= 0:
                continue
            x = [d['N'] for d in sub]
            y = [d['avg_time'] for d in sub]
            ax.plot(x, y, marker='<', linewidth=1.5, label='R={}, M={}'.format(R, M))
    ax.set_title('N vs. Average Computation Time')
    ax.set_xlabel('N')
    ax.set_ylabel('Average Computation Time (s)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_png


def plot_dslfmae_eval_table(summary_csv, output_png):
    rows = _read_csv_rows(summary_csv)
    if len(rows) == 0:
        return None

    rows = sorted(rows, key=lambda r: (
        _parse_int(r.get('M'), 0),
        _parse_int(r.get('R'), 0),
        _parse_int(r.get('N'), 0)
    ))

    col_labels = ['M', 'R', 'N', 'mean', 'std', 'Cb', 'Ca', 'Tcpu(s)']
    cell_text = []

    for r in rows:
        cell_text.append([
            str(r.get('M', '')),
            str(r.get('R', '')),
            str(r.get('N', '')),
            '{:.2f}'.format(_parse_float(r.get('mean_makespan'), 0.0)),
            '{:.2f}'.format(_parse_float(r.get('std_makespan'), 0.0)),
            '{:.2f}'.format(_parse_float(r.get('Cb'), 0.0)),
            '{:.2f}'.format(_parse_float(r.get('Ca'), 0.0)),
            '{:.4f}'.format(_parse_float(r.get('avg_cpu_time_sec'), 0.0))
        ])

    fig_h = max(6, 0.35 * len(cell_text) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis('off')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_png


def evaluate_saved_model_on_dslfmae_instances():
    """
    把当前 data 下 DSLFMAE 已解压算例，直接送给已训练好的模型评测。
    输出：
      1) 每次运行明细 CSV
      2) 每实例 all_runs.csv
      3) 每实例最佳甘特图 / 调度明细
      4) 汇总表 CSV
      5) 表格 PNG
      6) 平均计算时间三联图
    """
    ensure_dir(DSLFMAE_EVAL_DIR)

    model_path = DSLFMAE_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError('未找到 DSLFMAE 已训练模型: {}'.format(model_path))

    instances = load_dslfmae_instances(dataset_source=DSLFMAE_DATASET_SOURCE)

    all_runs_csv = os.path.join(DSLFMAE_EVAL_DIR, 'dslfmae_saved_model_all_runs.csv')
    summary_csv = os.path.join(DSLFMAE_EVAL_DIR, 'dslfmae_saved_model_summary.csv')
    table_csv = os.path.join(DSLFMAE_EVAL_DIR, 'dslfmae_saved_model_table.csv')
    table_png = os.path.join(DSLFMAE_EVAL_DIR, 'dslfmae_saved_model_table.png')
    avg_time_png = os.path.join(DSLFMAE_EVAL_DIR, 'dslfmae_saved_model_avg_computation_time.png')
    per_instance_dir = os.path.join(DSLFMAE_EVAL_DIR, 'per_instance')

    ensure_dir(per_instance_dir)

    agent = build_agent(initial_lr=INITIAL_LR)
    state_dict = torch.load(model_path, map_location=agent.device)
    load_policy_state_dict_compatible(agent, state_dict)
    agent.policy.eval()

    all_run_rows = []
    summary_rows = []

    print('\n==============================================================')
    print('开始评测 DSLFMAE 已训练模型')
    print('model_path    = {}'.format(model_path))
    print('dataset_root  = {}'.format(DSLFMAE_DATASET_SOURCE))
    print('eval_dir      = {}'.format(DSLFMAE_EVAL_DIR))
    print('runs          = {}'.format(DSLFMAE_EVAL_RUNS))
    print('stochastic    = {}'.format(DSLFMAE_EVAL_STOCHASTIC))
    print('==============================================================')

    for instance_id, instance_data in instances:
        instance_folder = os.path.join(per_instance_dir, instance_id)
        ensure_dir(instance_folder)

        cmax_list = []
        tcpu_list = []
        best_cmax = float('inf')
        best_res = None
        instance_run_rows = []

        M = int(instance_data.get('dslfmae_M', 0))
        R = int(instance_data.get('dslfmae_R', 0))
        N = int(instance_data.get('dslfmae_N', 0))

        print('\nRunning DSLFMAE instance: {} (M={}, R={}, N={})'.format(instance_id, M, R, N))

        for run_id in range(1, DSLFMAE_EVAL_RUNS + 1):
            res = run_policy_inference_once(
                agent=agent,
                instance_data=instance_data,
                ablation_cfg=ABLATION_VARIANTS['baseline'],
                stochastic=DSLFMAE_EVAL_STOCHASTIC
            )

            cmax = float(res['makespan'])
            tcpu = float(res['cpu_time'])

            cmax_list.append(cmax)
            tcpu_list.append(tcpu)

            run_row = {
                'instance_id': instance_id,
                'M': M,
                'R': R,
                'N': N,
                'run_id': int(run_id),
                'eval_mode': 'stochastic' if DSLFMAE_EVAL_STOCHASTIC else 'greedy',
                'makespan': cmax,
                'cpu_time_sec': tcpu
            }
            all_run_rows.append(run_row)
            instance_run_rows.append(run_row)

            if cmax < best_cmax:
                best_cmax = cmax
                best_res = copy.deepcopy(res)

            print('  run {:02d}/{} | makespan = {:.2f} | cpu = {:.4f}s'.format(
                run_id, DSLFMAE_EVAL_RUNS, cmax, tcpu
            ))

        mean_cmax = float(np.mean(cmax_list))
        std_cmax = float(np.std(cmax_list, ddof=0))
        cb = float(np.min(cmax_list))
        ca = float(np.mean(cmax_list))
        avg_tcpu = float(np.mean(tcpu_list))

        instance_runs_csv = os.path.join(instance_folder, 'all_runs.csv')
        save_csv_rows(instance_run_rows, instance_runs_csv)

        best_schedule_csv = export_schedule_detail(
            best_res['schedule_log'],
            save_dir=instance_folder,
            filename='best_schedule_detail.csv',
            machine_labels=best_res['machine_labels'],
            original_machine_labels=best_res['original_machine_labels'],
            compact_to_original_machine_id=best_res['compact_to_original_machine_id']
        )

        best_gantt_png = save_gantt_chart(
            best_res['schedule_log'],
            best_res['num_machines'],
            best_res['makespan'],
            save_dir=instance_folder,
            filename='best_gantt.png',
            machine_labels=best_res['machine_labels'],
            drop_idle_machines=False
        )

        summary_rows.append({
            'instance_id': instance_id,
            'M': M,
            'R': R,
            'N': N,
            'runs': int(DSLFMAE_EVAL_RUNS),
            'eval_mode': 'stochastic' if DSLFMAE_EVAL_STOCHASTIC else 'greedy',
            'mean_makespan': mean_cmax,
            'std_makespan': std_cmax,
            'Cb': cb,
            'Ca': ca,
            'avg_cpu_time_sec': avg_tcpu,
            'all_runs_csv': instance_runs_csv,
            'best_schedule_csv': best_schedule_csv,
            'best_gantt_png': best_gantt_png
        })

        print('  [SUMMARY] mean = {:.2f} | std = {:.2f} | Cb = {:.2f} | Ca = {:.2f} | Tcpu = {:.4f}s'.format(
            mean_cmax, std_cmax, cb, ca, avg_tcpu
        ))

    summary_rows = sorted(summary_rows, key=lambda x: (int(x['M']), int(x['R']), int(x['N'])))
    save_csv_rows(all_run_rows, all_runs_csv)
    save_csv_rows(summary_rows, summary_csv)

    # 再导出一个更接近表格格式的精简 CSV
    table_rows = []
    for r in summary_rows:
        table_rows.append({
            'M': r['M'],
            'R': r['R'],
            'N': r['N'],
            'mean': r['mean_makespan'],
            'std': r['std_makespan'],
            'Cb': r['Cb'],
            'Ca': r['Ca'],
            'Tcpu_sec': r['avg_cpu_time_sec']
        })
    save_csv_rows(table_rows, table_csv)

    plot_dslfmae_eval_table(summary_csv, table_png)
    plot_dslfmae_avg_computation_time_from_eval_summary(summary_csv, avg_time_png)

    print('\n✅ DSLFMAE 已训练模型评测完成')
    print('每次运行明细: {}'.format(all_runs_csv))
    print('汇总结果    : {}'.format(summary_csv))
    print('表格 CSV    : {}'.format(table_csv))
    print('表格 PNG    : {}'.format(table_png))
    print('时间图      : {}'.format(avg_time_png))


def main():
    if MODE == 'train_round_robin':
        train_round_robin_24cases()

    elif MODE == 'eval_best_model':
        evaluate_saved_model_on_public_instances()

    elif MODE == 'eval_industrial':
        evaluate_saved_model_on_industrial_orders()

    elif MODE == 'train_and_eval':
        train_round_robin_24cases()
        evaluate_saved_model_on_public_instances()

    elif MODE == 'train_single_variant':
        train_single_variant()

    elif MODE == 'eval_single_variant':
        eval_single_variant()

    elif MODE == 'train_and_eval_single':
        train_and_eval_single()

    elif MODE == 'run_public_ablation_suite':
        run_public_ablation_suite()

    elif MODE == 'run_dslfmae_table13':
        run_dslfmae_table13()

    elif MODE == 'eval_dslfmae_table13':
        evaluate_saved_model_on_dslfmae_instances()

    else:
        raise ValueError('未知 MODE: {}'.format(MODE))


if __name__ == "__main__":
    main()