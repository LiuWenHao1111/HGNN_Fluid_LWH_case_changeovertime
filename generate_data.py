import os
import re
import csv
import ast
import json
import copy
import numpy as np
import  glob


def generate_random_instance(is_validation=False, seed=None):
    """按照原训练逻辑生成随机实例"""
    if seed is not None:
        np.random.seed(seed)

    if is_validation:
        M, R = 7, 8
    else:
        M = np.random.randint(3, 9)
        R = np.random.randint(3, 13)

    job_types = []
    for r in range(R):
        Jr = np.random.randint(3, 6)
        Nr = np.random.randint(5, 51)

        ops = []
        for j in range(Jr):
            num_compat = np.random.randint(2, M + 1) if M >= 2 else 1
            compat_m = np.random.choice(range(M), num_compat, replace=False)

            proc_times = {}
            for m in compat_m:
                proc_times[int(m)] = int(np.random.randint(40, 401))

            ops.append({
                'op_id': int(j),
                'compatible_machines': [int(x) for x in sorted(list(compat_m))],
                'processing_times': proc_times
            })

        job_types.append({
            'type_id': int(r),
            'num_jobs': int(Nr),
            'ops': ops
        })

    return {
        'num_job_types': int(R),
        'num_machines': int(M),
        'job_types': job_types
    }


STANDARD_INSTANCE_IDS = [
    'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
    'P7', 'P8', 'P9', 'P10', 'P11', 'P12',
    'P13', 'P14', 'P15', 'P16', 'P17', 'P18',
    'P19', 'P20', 'P21', 'P22', 'P23', 'P24'
]

RAW_TO_STANDARD_MAP = {
    'P11': 'P1',  'P12': 'P2',  'P13': 'P3',
    'P21': 'P4',  'P22': 'P5',  'P23': 'P6',
    'P31': 'P7',  'P32': 'P8',  'P33': 'P9',
    'P41': 'P10', 'P42': 'P11', 'P43': 'P12',
    'P51': 'P13', 'P52': 'P14', 'P53': 'P15',
    'P61': 'P16', 'P62': 'P17', 'P63': 'P18',
    'P71': 'P19', 'P72': 'P20', 'P73': 'P21',
    'P81': 'P22', 'P82': 'P23', 'P83': 'P24'
}


def _parse_sequence(text):
    value = ast.literal_eval(str(text))
    if isinstance(value, (tuple, list)):
        return [int(x) for x in value]
    return [int(value)]


def _parse_nested_numeric_matrix(text):
    value = ast.literal_eval(str(text))
    if not isinstance(value, (tuple, list)):
        raise ValueError('changeover 矩阵必须是二维 tuple/list: {}'.format(text))
    matrix = []
    for row in value:
        if not isinstance(row, (tuple, list)):
            raise ValueError('changeover 矩阵每一行必须是 tuple/list: {}'.format(text))
        matrix.append([float(x) for x in row])
    return matrix


def _read_changeover_kind_matrix(instance_dir, kind_count, machine_count):
    """
    读取本次上传的数据文件：changeover_kind_matrix_by_machine.csv

    文件格式：
      machine,kind_changeover_matrix,kind_order
      0,"((7, 6, 9),(7, 7, 6),(9, 7, 7))","(0, 1, 2)"

    返回：
      {
        machine_id: {
          'kind_order': [0,1,2],
          'matrix': [[...], [...], [...]]
        },
        ...
      }

    matrix[i][j] 表示同一机器上，从 kind_order[i] 切换到 kind_order[j] 的切换时间。
    """
    path = os.path.join(instance_dir, 'changeover_kind_matrix_by_machine.csv')
    if not os.path.exists(path):
        return {}, False

    result = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'machine' not in row:
                raise KeyError('changeover_kind_matrix_by_machine.csv 缺少 machine 字段')
            if 'kind_changeover_matrix' not in row:
                raise KeyError('changeover_kind_matrix_by_machine.csv 缺少 kind_changeover_matrix 字段')
            if 'kind_order' not in row:
                raise KeyError('changeover_kind_matrix_by_machine.csv 缺少 kind_order 字段')

            machine = int(row['machine'])
            kind_order = _parse_sequence(row['kind_order'])
            matrix = _parse_nested_numeric_matrix(row['kind_changeover_matrix'])

            if len(kind_order) != kind_count:
                raise ValueError(
                    'machine {} 的 kind_order 长度 {} 与 kind_count {} 不一致'.format(
                        machine, len(kind_order), kind_count
                    )
                )
            if len(matrix) != len(kind_order):
                raise ValueError('machine {} 的 changeover 矩阵行数与 kind_order 不一致'.format(machine))
            for line in matrix:
                if len(line) != len(kind_order):
                    raise ValueError('machine {} 的 changeover 矩阵列数与 kind_order 不一致'.format(machine))

            result[machine] = {
                'kind_order': [int(x) for x in kind_order],
                'matrix': matrix
            }

    # 没有在 changeover 文件中出现的机器，补全 0 矩阵，防止后续查找报错。
    zero_matrix = [[0.0 for _ in range(kind_count)] for _ in range(kind_count)]
    default_order = [int(i) for i in range(kind_count)]
    for m in range(machine_count):
        if m not in result:
            result[m] = {
                'kind_order': default_order[:],
                'matrix': copy.deepcopy(zero_matrix)
            }

    return result, True


def _normalize_changeover_matrix_dict(matrix_dict):
    norm = {}
    for m, item in matrix_dict.items():
        norm[int(m)] = {
            'kind_order': [int(x) for x in item.get('kind_order', [])],
            'matrix': [[float(v) for v in row] for row in item.get('matrix', [])]
        }
    return norm


def _normalize_instance_data(data):
    norm = {
        'instance_name': data.get('instance_name', None),
        'num_job_types': int(data['num_job_types']),
        'num_machines': int(data['num_machines']),
        'job_types': []
    }

    for k, v in data.items():
        if k not in ['instance_name', 'num_job_types', 'num_machines', 'job_types']:
            if k == 'changeover_kind_matrix_by_machine':
                norm[k] = _normalize_changeover_matrix_dict(v)
            else:
                norm[k] = v

    for jt in data['job_types']:
        new_jt = {
            'type_id': int(jt['type_id']),
            'num_jobs': int(jt['num_jobs']),
            'ops': []
        }

        for extra_key in jt.keys():
            if extra_key not in ['type_id', 'num_jobs', 'ops']:
                new_jt[extra_key] = jt[extra_key]

        for op in jt['ops']:
            proc_times = {}
            for k, v in op['processing_times'].items():
                proc_times[int(k)] = int(v)

            new_op = {
                'op_id': int(op['op_id']),
                'compatible_machines': [int(m) for m in op['compatible_machines']],
                'processing_times': proc_times
            }

            for extra_key in op.keys():
                if extra_key not in ['op_id', 'compatible_machines', 'processing_times']:
                    new_op[extra_key] = op[extra_key]

            new_jt['ops'].append(new_op)

        norm['job_types'].append(new_jt)

    return norm


def load_instance_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return _normalize_instance_data(data)


def load_instance_from_csv_folder(folder_path, instance_name=None):
    """
    兼容 CSV 目录：
      based_data.csv
      order_data.csv
      process_data.csv

    支持字段：
      order: kind_number / kind_count
      process: machines / machine_selectable
               times / process_time

    注意：这是公开/标准/DSLFMAE 实例读取函数，不启用工业 changeover 特征。
    工业订单读取使用 load_industrial_orders()。
    """
    based_path = os.path.join(folder_path, 'based_data.csv')
    order_path = os.path.join(folder_path, 'order_data.csv')
    process_path = os.path.join(folder_path, 'process_data.csv')

    if not os.path.exists(based_path):
        raise FileNotFoundError('missing based_data.csv: {}'.format(folder_path))
    if not os.path.exists(order_path):
        raise FileNotFoundError('missing order_data.csv: {}'.format(folder_path))
    if not os.path.exists(process_path):
        raise FileNotFoundError('missing process_data.csv: {}'.format(folder_path))

    with open(based_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) == 0:
            raise ValueError('empty based_data.csv: {}'.format(folder_path))
        row = rows[0]

    kind_count = int(row['kind_count'])
    machine_count = int(row['machine_count'])

    kind_job_counts = [0 for _ in range(kind_count)]
    with open(order_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qty_field = None
            if 'kind_number' in row:
                qty_field = 'kind_number'
            elif 'kind_count' in row:
                qty_field = 'kind_count'
            else:
                raise KeyError('order_data.csv 缺少 kind_number / kind_count 字段: {}'.format(folder_path))

            nums = _parse_sequence(row[qty_field])
            if len(nums) != kind_count:
                raise ValueError('订单中的工件种类数量与 based_data 不一致: {}'.format(folder_path))

            for i in range(kind_count):
                kind_job_counts[i] += int(nums[i])

    ops_by_kind = {}
    for k in range(kind_count):
        ops_by_kind[k] = []

    with open(process_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = int(row['kind'])
            task = int(row['task'])

            if 'machines' in row:
                machines = _parse_sequence(row['machines'])
            elif 'machine_selectable' in row:
                machines = _parse_sequence(row['machine_selectable'])
            else:
                raise KeyError('process_data.csv 缺少 machines / machine_selectable 字段: {}'.format(folder_path))

            if 'times' in row:
                times = _parse_sequence(row['times'])
            elif 'process_time' in row:
                times = _parse_sequence(row['process_time'])
            else:
                raise KeyError('process_data.csv 缺少 times / process_time 字段: {}'.format(folder_path))

            if len(machines) != len(times):
                raise ValueError('machine/time length mismatch in {}'.format(folder_path))

            proc_times = {}
            for m, t in zip(machines, times):
                proc_times[int(m)] = int(t)

            ops_by_kind[kind].append({
                'op_id': int(task),
                'compatible_machines': [int(m) for m in machines],
                'processing_times': proc_times
            })

    job_types = []
    for kind in range(kind_count):
        ops_sorted = sorted(ops_by_kind[kind], key=lambda x: x['op_id'])
        job_types.append({
            'type_id': int(kind),
            'num_jobs': int(kind_job_counts[kind]),
            'ops': ops_sorted
        })

    data = {
        'instance_name': instance_name,
        'num_job_types': int(kind_count),
        'num_machines': int(machine_count),
        'job_types': job_types
    }
    return _normalize_instance_data(data)


def _detect_raw_folder_mode(data_dir):
    found_raw = []
    for raw_id in RAW_TO_STANDARD_MAP.keys():
        if os.path.isdir(os.path.join(data_dir, raw_id)) or \
           os.path.exists(os.path.join(data_dir, raw_id + '.json')) or \
           os.path.exists(os.path.join(data_dir, raw_id + '(1).json')):
            found_raw.append(raw_id)
    return len(found_raw) > 0, found_raw


def load_all_standard_instances(data_dir='data'):
    instances = {}

    raw_mode, raw_found = _detect_raw_folder_mode(data_dir)

    if raw_mode:
        print('检测到原始目录格式实例，启用映射读取模式。')
        print('原始目录: {}'.format(', '.join(sorted(raw_found))))

        for raw_id, std_id in RAW_TO_STANDARD_MAP.items():
            folder = os.path.join(data_dir, raw_id)
            if os.path.isdir(folder):
                instances[std_id] = load_instance_from_csv_folder(folder, instance_name=std_id)
                continue

            json_candidates = [
                os.path.join(data_dir, raw_id + '.json'),
                os.path.join(data_dir, raw_id + '(1).json'),
                os.path.join(data_dir, raw_id.lower() + '.json'),
                os.path.join(data_dir, raw_id.lower() + '(1).json')
            ]
            found = False
            for path in json_candidates:
                if os.path.exists(path):
                    instances[std_id] = load_instance_from_json(path)
                    instances[std_id]['instance_name'] = std_id
                    found = True
                    break

            if not found:
                raise FileNotFoundError('未找到原始实例 {} 对应的数据'.format(raw_id))

    else:
        print('未检测到原始目录格式实例，启用标准 P1~P24 直接读取模式。')

        for pid in STANDARD_INSTANCE_IDS:
            json_candidates = [
                os.path.join(data_dir, pid + '.json'),
                os.path.join(data_dir, pid + '(1).json'),
                os.path.join(data_dir, pid.lower() + '.json'),
                os.path.join(data_dir, pid.lower() + '(1).json')
            ]

            loaded = False
            for path in json_candidates:
                if os.path.exists(path):
                    instances[pid] = load_instance_from_json(path)
                    instances[pid]['instance_name'] = pid
                    loaded = True
                    break

            if loaded:
                continue

            folder = os.path.join(data_dir, pid)
            if os.path.isdir(folder):
                instances[pid] = load_instance_from_csv_folder(folder, instance_name=pid)
                loaded = True

            if not loaded:
                raise FileNotFoundError('未找到标准实例 {}'.format(pid))

    missing = [pid for pid in STANDARD_INSTANCE_IDS if pid not in instances]
    if missing:
        raise FileNotFoundError('缺少标准算例: {}'.format(', '.join(missing)))

    ordered = []
    for pid in STANDARD_INSTANCE_IDS:
        ordered.append((pid, instances[pid]))

    return ordered


def load_standard_instance_by_id(instance_id, data_dir='data'):
    all_instances = dict(load_all_standard_instances(data_dir))
    if instance_id not in all_instances:
        raise KeyError('unknown instance_id: {}'.format(instance_id))
    return all_instances[instance_id]


# =========================================================
# 工业订单实例：每个订单单独构造成一个实例
# =========================================================
def _read_industrial_process_template(instance_dir):
    based_path = os.path.join(instance_dir, 'based_data.csv')
    process_path = os.path.join(instance_dir, 'process_data.csv')

    if not os.path.exists(based_path):
        raise FileNotFoundError('missing based_data.csv: {}'.format(instance_dir))
    if not os.path.exists(process_path):
        raise FileNotFoundError('missing process_data.csv: {}'.format(instance_dir))

    with open(based_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) == 0:
            raise ValueError('empty based_data.csv: {}'.format(instance_dir))
        row = rows[0]

    kind_count = int(row['kind_count'])
    machine_count = int(row['machine_count'])

    changeover_kind_matrix_by_machine, has_changeover_matrix = _read_changeover_kind_matrix(
        instance_dir=instance_dir,
        kind_count=kind_count,
        machine_count=machine_count
    )

    ops_by_kind = {}
    for k in range(kind_count):
        ops_by_kind[k] = []

    with open(process_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = int(row['kind'])
            task = int(row['task'])

            if 'machines' in row:
                machines = _parse_sequence(row['machines'])
            elif 'machine_selectable' in row:
                machines = _parse_sequence(row['machine_selectable'])
            else:
                raise KeyError('process_data.csv 缺少 machines / machine_selectable 字段: {}'.format(instance_dir))

            if 'times' in row:
                times = _parse_sequence(row['times'])
            elif 'process_time' in row:
                times = _parse_sequence(row['process_time'])
            else:
                raise KeyError('process_data.csv 缺少 times / process_time 字段: {}'.format(instance_dir))

            if len(machines) != len(times):
                raise ValueError('machine/time length mismatch in {}'.format(instance_dir))

            proc_times = {}
            for m, t in zip(machines, times):
                proc_times[int(m)] = int(t)

            ops_by_kind[kind].append({
                'op_id': int(task),
                'compatible_machines': [int(m) for m in machines],
                'processing_times': proc_times
            })

    for k in ops_by_kind:
        ops_by_kind[k] = sorted(ops_by_kind[k], key=lambda x: x['op_id'])

    return kind_count, machine_count, ops_by_kind, changeover_kind_matrix_by_machine, has_changeover_matrix


def _remap_changeover_matrix_by_machine(changeover_kind_matrix_by_machine, compact_to_original_machine_id):
    """机器压缩编号后，把 changeover 矩阵也从原始机器号映射到压缩机器号。"""
    remapped = {}
    for compact_m, original_m in enumerate(compact_to_original_machine_id):
        if original_m in changeover_kind_matrix_by_machine:
            remapped[int(compact_m)] = copy.deepcopy(changeover_kind_matrix_by_machine[original_m])
    return remapped


def _remap_active_machines_contiguous(raw_job_types, active_machine_ids):
    sorted_original_ids = sorted(list(active_machine_ids))
    if len(sorted_original_ids) == 0:
        raise ValueError('当前订单没有活跃机器，无法构造实例。')

    old_to_new = {}
    for new_id, old_id in enumerate(sorted_original_ids):
        old_to_new[old_id] = new_id

    compact_machine_labels = ['M{}'.format(i + 1) for i in range(len(sorted_original_ids))]
    original_machine_labels = ['M{}'.format(old_id + 1) for old_id in sorted_original_ids]

    new_job_types = []
    for jt in raw_job_types:
        new_ops = []
        for op in jt['ops']:
            new_proc = {}
            new_compat = []

            for old_m in op['compatible_machines']:
                new_m = old_to_new[old_m]
                new_compat.append(new_m)
                new_proc[new_m] = op['processing_times'][old_m]

            new_op = copy.deepcopy(op)
            new_op['compatible_machines'] = sorted(new_compat)
            new_op['processing_times'] = new_proc
            new_ops.append(new_op)

        new_jt = copy.deepcopy(jt)
        new_jt['ops'] = new_ops
        new_job_types.append(new_jt)

    compact_to_original_machine_id = sorted_original_ids

    return new_job_types, compact_machine_labels, original_machine_labels, compact_to_original_machine_id


def load_industrial_orders(instance_dir='data/instance',
                           compact_unused_machines=True,
                           drop_zero_job_types=True):
    order_path = os.path.join(instance_dir, 'order_data.csv')
    if not os.path.exists(order_path):
        raise FileNotFoundError('missing order_data.csv: {}'.format(instance_dir))

    kind_count, machine_count, ops_by_kind, changeover_kind_matrix_by_machine, has_changeover_matrix = \
        _read_industrial_process_template(instance_dir)

    order_instances = []
    with open(order_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        order_rows = list(reader)

    for row_idx, row in enumerate(order_rows):
        if 'kind_count' in row:
            counts = _parse_sequence(row['kind_count'])
        elif 'kind_number' in row:
            counts = _parse_sequence(row['kind_number'])
        else:
            raise KeyError('order_data.csv 缺少 kind_count / kind_number 字段: {}'.format(instance_dir))

        if len(counts) != kind_count:
            raise ValueError('订单工件种类数与 based_data 不一致: {}'.format(instance_dir))

        selected_kind_ids = []
        for k in range(kind_count):
            if drop_zero_job_types:
                if int(counts[k]) > 0:
                    selected_kind_ids.append(k)
            else:
                selected_kind_ids.append(k)

        raw_job_types = []
        used_machine_ids = set()

        for new_type_id, old_kind_id in enumerate(selected_kind_ids):
            ops = copy.deepcopy(ops_by_kind[old_kind_id])

            for op in ops:
                used_machine_ids.update(op['compatible_machines'])

            raw_job_types.append({
                'type_id': int(new_type_id),
                'original_type_id': int(old_kind_id),
                'num_jobs': int(counts[old_kind_id]),
                'ops': ops
            })

        if compact_unused_machines:
            remapped_job_types, compact_machine_labels, original_machine_labels, compact_to_original_machine_id = \
                _remap_active_machines_contiguous(raw_job_types, used_machine_ids)
            num_machines = len(compact_machine_labels)
            active_changeover_matrix = _remap_changeover_matrix_by_machine(
                changeover_kind_matrix_by_machine,
                compact_to_original_machine_id
            )
        else:
            remapped_job_types = raw_job_types
            compact_machine_labels = ['M{}'.format(i + 1) for i in range(machine_count)]
            original_machine_labels = ['M{}'.format(i + 1) for i in range(machine_count)]
            compact_to_original_machine_id = list(range(machine_count))
            num_machines = int(machine_count)
            active_changeover_matrix = copy.deepcopy(changeover_kind_matrix_by_machine)

        data = {
            'instance_name': 'Order{}'.format(row_idx + 1),
            'num_job_types': int(len(remapped_job_types)),
            'num_machines': int(num_machines),
            'job_types': remapped_job_types,
            'machine_labels': compact_machine_labels,
            'original_machine_labels': original_machine_labels,
            'compact_to_original_machine_id': compact_to_original_machine_id,
            'order_kind_counts': [int(x) for x in counts],

            # 工业实例专用标记与数据。
            'is_industrial_instance': True,
            'use_changeover_feature': bool(has_changeover_matrix),
            'has_changeover_data': bool(has_changeover_matrix),
            'changeover_feature_dim': 1,
            'changeover_kind_matrix_by_machine': active_changeover_matrix
        }

        order_instances.append((data['instance_name'], _normalize_instance_data(data)))

    return order_instances


# =========================================================
# DSLFMAE 数据集：只读取 data 下已解压目录
# 支持：
#   data/DSLFMAE/M10R5N5
# 或
#   data/M10R5N5
# =========================================================
def _parse_mrn_from_folder_name(name):
    m = re.match(r'^M(\d+)R(\d+)N(\d+)$', name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _count_mrn_dirs(folder):
    count = 0
    if not os.path.isdir(folder):
        return 0
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isdir(full) and _parse_mrn_from_folder_name(name) is not None:
            count += 1
    return count


def _resolve_dslfmae_root(dataset_source='data'):
    """
    只从已解压目录读取，不再解压 zip。

    支持两种结构：
    1) data/DSLFMAE/M10R5N5
    2) data/M10R5N5
    """
    if not os.path.isdir(dataset_source):
        raise FileNotFoundError('DSLFMAE 数据目录不存在: {}'.format(dataset_source))

    # 情况 1：data 下直接就是 M10R5N5, M10R5N10, ...
    direct_count = _count_mrn_dirs(dataset_source)
    if direct_count > 0:
        return dataset_source

    # 情况 2：data/DSLFMAE/ 下才是实例目录
    preferred_subdirs = ['DSLFMAE', 'dslfmae']
    for sub in preferred_subdirs:
        subdir = os.path.join(dataset_source, sub)
        if _count_mrn_dirs(subdir) > 0:
            return subdir

    # 情况 3：data 下某个其他子目录里才是实例
    best_subdir = None
    best_count = 0
    for name in os.listdir(dataset_source):
        subdir = os.path.join(dataset_source, name)
        if not os.path.isdir(subdir):
            continue
        c = _count_mrn_dirs(subdir)
        if c > best_count:
            best_count = c
            best_subdir = subdir

    if best_subdir is not None and best_count > 0:
        return best_subdir

    raise FileNotFoundError(
        '在 {} 下未找到 DSLFMAE 已解压实例目录（例如 M10R5N5）'.format(dataset_source)
    )


def load_dslfmae_instances(dataset_source='data'):
    """
    读取 27 个 DSLFMAE 实例，按 (M, R, N) 排序返回：
      [
        ('M10R5N5', data),
        ...
      ]
    """
    root_dir = _resolve_dslfmae_root(dataset_source)

    folder_names = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and _parse_mrn_from_folder_name(name) is not None:
            folder_names.append(name)

    if len(folder_names) == 0:
        raise FileNotFoundError('在 {} 中没有找到 DSLFMAE 实例目录'.format(root_dir))

    folder_names = sorted(folder_names, key=lambda x: _parse_mrn_from_folder_name(x))

    instances = []
    for folder_name in folder_names:
        mrn = _parse_mrn_from_folder_name(folder_name)
        M, R, N = mrn
        folder_path = os.path.join(root_dir, folder_name)

        data = load_instance_from_csv_folder(folder_path, instance_name=folder_name)
        data['dslfmae_M'] = int(M)
        data['dslfmae_R'] = int(R)
        data['dslfmae_N'] = int(N)

        instances.append((folder_name, _normalize_instance_data(data)))

    return instances
