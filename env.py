import numpy as np
from fluid_model import FluidModel


class FJSPEnv(object):
    def __init__(self, data):
        self.data = data
        self.num_machines = int(data['num_machines'])
        self.num_job_types = int(data['num_job_types'])
        self.fluid_solver = FluidModel(data)
        self.total_jobs_init = max(1, sum([jt['num_jobs'] for jt in self.data['job_types']]))
        self.schedule_log = []

    # =========================================================
    # 工业实例 changeover 特征相关工具
    # =========================================================
    def _use_changeover_feature(self):
        """
        只在工业实例上启用 changeover 特征。
        普通随机实例、公开标准实例、DSLFMAE 实例不设置该标记，因此仍保持 edge_dim=6。
        """
        return bool(self.data.get('is_industrial_instance', False)) and \
            bool(self.data.get('use_changeover_feature', False))

    def _get_edge_dim(self):
        return 7 if self._use_changeover_feature() else 6

    def _get_original_kind_id(self, r):
        """
        工业订单中 drop_zero_job_types=True 时，内部 type_id 可能被重新编号。
        changeover_kind_matrix_by_machine.csv 使用的是原始 kind 编号，因此这里优先取 original_type_id。
        """
        jt = self.data['job_types'][int(r)]
        return int(jt.get('original_type_id', jt.get('type_id', r)))

    def _get_machine_reference_kind_for_changeover(self, m):
        """
        返回机器 m 当前用于计算切换时间的前序产品类型：
        - 机器忙：用正在加工的工件类型，表示当前加工结束后再切到候选类型；
        - 机器空闲：用该机器最近一次完成加工的工件类型；
        - 机器从未加工过：返回 None，此时切换时间为 0。
        """
        machine = self.machines[int(m)]
        if machine.get('current_op', None) is not None:
            r_cur, _ = machine['current_op']
            return self._get_original_kind_id(r_cur)
        return machine.get('last_kind', None)

    def _get_changeover_time(self, r, j, m):
        """
        根据上传的 changeover_kind_matrix_by_machine.csv 计算弧 (o_rj, m) 的切换时间特征。

        数据格式：
            machine,kind_changeover_matrix,kind_order
            0,"((7, 6, 9),(7, 7, 6),(9, 7, 7))","(0, 1, 2)"

        解释：
            matrix[previous_kind][current_kind]
            previous_kind 为机器 m 上一个/当前加工的原始产品类型；
            current_kind 为候选工序所属的原始产品类型。

        如果机器尚未加工过任何产品，则不发生切换，返回 0。
        """
        if not self._use_changeover_feature():
            return 0.0

        prev_kind = self._get_machine_reference_kind_for_changeover(m)
        if prev_kind is None:
            return 0.0

        cur_kind = self._get_original_kind_id(r)
        matrix_info = self.data.get('changeover_kind_matrix_by_machine', {})

        # 兼容 int key / str key。
        item = None
        for key in [int(m), str(int(m))]:
            if key in matrix_info:
                item = matrix_info[key]
                break
        if item is None:
            return 0.0

        matrix = item.get('matrix', [])
        kind_order = item.get('kind_order', [])
        kind_to_pos = {int(k): idx for idx, k in enumerate(kind_order)}

        if int(prev_kind) not in kind_to_pos or int(cur_kind) not in kind_to_pos:
            return 0.0

        i = kind_to_pos[int(prev_kind)]
        k = kind_to_pos[int(cur_kind)]

        try:
            return float(matrix[i][k])
        except Exception:
            return 0.0

    def reset(self):
        self.current_time = 0.0
        self.finished_jobs = 0
        self.schedule_log = []

        self.queues = {}
        self.Q_plus = {}
        self.Q_minus_machine = {}
        for m in range(self.num_machines):
            self.Q_minus_machine[m] = {}

        for r, jt in enumerate(self.data['job_types']):
            n_jobs = int(jt['num_jobs'])
            for op in jt['ops']:
                j = int(op['op_id'])
                self.queues[(r, j)] = n_jobs if j == 0 else 0
                self.Q_plus[(r, j)] = n_jobs
                for m in range(self.num_machines):
                    self.Q_minus_machine[m][(r, j)] = 0

        self.machines = {}
        for m in range(self.num_machines):
            self.machines[m] = {
                'status': 0,
                'finish_time': 0.0,
                'current_op': None,
                'start_time': 0.0,
                'last_kind': None,
                'current_changeover_time': 0.0
            }

        self.initial_Q_plus = self.Q_plus.copy()
        self.u_mrj, self.C_max_f = self.fluid_solver.solve(self.initial_Q_plus)
        self.C_max_f = max(float(self.C_max_f), 1.0)
        self.fluid_snapshot = self.fluid_solver.get_fluid_state(
            self.u_mrj,
            self.initial_Q_plus,
            self.current_time
        )

        self.last_C_t = self._compute_C_t()
        return self._get_state()

    def _get_op_idx(self, r, j):
        return self.fluid_solver.op_to_idx[(r, j)]

    def _get_op_from_idx(self, idx):
        return self.fluid_solver.op_list[idx]

    def _compute_C_t(self):
        candidates = [self.current_time]

        for m in range(self.num_machines):
            if self.machines[m]['status'] == 1:
                candidates.append(float(self.machines[m]['finish_time']))

        for (r, j), q_left in self.Q_plus.items():
            if q_left <= 0:
                continue
            e_rj = float(self.fluid_snapshot['e_rj'].get((r, j), 0.0))
            if e_rj > 1e-8:
                candidates.append(self.current_time + float(q_left) / e_rj)
            else:
                compat = [int(m) for m in self.data['job_types'][r]['ops'][j]['compatible_machines']]
                min_t = min([self.data['job_types'][r]['ops'][j]['processing_times'][m] for m in compat])
                candidates.append(self.current_time + float(q_left) * float(min_t))

        return float(max(candidates))

    def step(self, action):
        r, j = self._get_op_from_idx(action[0])
        m = int(action[1])

        if self.queues[(r, j)] <= 0:
            raise ValueError('Invalid action: empty queue for ({}, {}).'.format(r, j))
        if self.machines[m]['status'] != 0:
            raise ValueError('Invalid action: machine {} is busy.'.format(m))

        self.queues[(r, j)] -= 1
        proc_time = float(self.data['job_types'][r]['ops'][j]['processing_times'][m])

        # 注意：这里按你的要求只把 changeover 作为工业实例的图特征。
        # 不把 changeover 加入实际加工完成时间，避免改变原奖励、完工时间和仿真推进逻辑。
        changeover_time = self._get_changeover_time(r, j, m)

        self.machines[m]['status'] = 1
        self.machines[m]['finish_time'] = self.current_time + proc_time
        self.machines[m]['current_op'] = (r, j)
        self.machines[m]['start_time'] = self.current_time
        self.machines[m]['current_changeover_time'] = float(changeover_time)

        self.Q_minus_machine[m][(r, j)] += 1
        self.Q_plus[(r, j)] = max(0, self.Q_plus[(r, j)] - 1)

        self._advance_time()
        self.fluid_snapshot = self.fluid_solver.get_fluid_state(
            self.u_mrj,
            self.initial_Q_plus,
            self.current_time
        )
        current_C_t = self._compute_C_t()

        reward = (self.last_C_t - current_C_t) / self.C_max_f
        done = (self.finished_jobs == self.total_jobs_init)
        self.last_C_t = current_C_t

        return self._get_state(), float(reward), done, {}

    def _advance_time(self):
        while True:
            if self.finished_jobs == self.total_jobs_init:
                return
            if len(self.get_valid_actions()) > 0:
                return

            busy_machines = [m for m in self.machines if self.machines[m]['status'] == 1]
            if not busy_machines:
                if sum(self.queues.values()) == 0:
                    self.finished_jobs = self.total_jobs_init
                return

            min_ft = min([self.machines[m]['finish_time'] for m in busy_machines])
            self.current_time = float(min_ft)

            for m in [x for x in busy_machines if abs(self.machines[x]['finish_time'] - min_ft) <= 1e-9]:
                r, j = self.machines[m]['current_op']
                original_kind = self._get_original_kind_id(r)
                changeover_time = float(self.machines[m].get('current_changeover_time', 0.0))

                self.schedule_log.append({
                    'machine': m,
                    'type': r,
                    'original_type': original_kind,
                    'op': j,
                    'start': self.machines[m]['start_time'],
                    'end': self.current_time,
                    'changeover_time_feature': changeover_time
                })
                self.machines[m]['status'] = 0
                self.machines[m]['current_op'] = None
                self.machines[m]['last_kind'] = original_kind
                self.machines[m]['current_changeover_time'] = 0.0

                if j + 1 < len(self.data['job_types'][r]['ops']):
                    next_op_id = int(self.data['job_types'][r]['ops'][j + 1]['op_id'])
                    self.queues[(r, next_op_id)] = self.queues.get((r, next_op_id), 0) + 1
                else:
                    self.finished_jobs += 1

    def get_valid_actions(self):
        actions = []
        for r_idx, jt in enumerate(self.data['job_types']):
            for op in jt['ops']:
                j = int(op['op_id'])
                if self.queues.get((r_idx, j), 0) > 0:
                    for m in op['compatible_machines']:
                        m = int(m)
                        if self.machines[m]['status'] == 0 and self.u_mrj.get((m, r_idx, j), 0.0) > 1e-9:
                            actions.append((self._get_op_idx(r_idx, j), m))
        return actions

    def _build_op_edges(self):
        edge_index_prev = []
        edge_index_next = []
        for r in range(self.num_job_types):
            ops = self.data['job_types'][r]['ops']
            for k in range(len(ops) - 1):
                cur_idx = self._get_op_idx(r, int(ops[k]['op_id']))
                nxt_idx = self._get_op_idx(r, int(ops[k + 1]['op_id']))
                edge_index_prev.append([nxt_idx, cur_idx])
                edge_index_next.append([cur_idx, nxt_idx])

        if edge_index_prev:
            edge_index_prev = np.array(edge_index_prev, dtype=np.int64).T
        else:
            edge_index_prev = np.zeros((2, 0), dtype=np.int64)

        if edge_index_next:
            edge_index_next = np.array(edge_index_next, dtype=np.int64).T
        else:
            edge_index_next = np.zeros((2, 0), dtype=np.int64)

        return edge_index_prev, edge_index_next

    def _get_state(self):
        snapshot = self.fluid_snapshot
        norm_time = max(self.C_max_f, 1.0)
        norm_q = float(self.total_jobs_init)

        x_m, x_o, edge_index_om, edge_attr_om = [], [], [], []

        for m in range(self.num_machines):
            compat_ops = []
            for r in range(self.num_job_types):
                for op in self.data['job_types'][r]['ops']:
                    j = int(op['op_id'])
                    if m in op['compatible_machines']:
                        compat_ops.append((r, j))

            deg_m = len(compat_ops)
            mean_t = np.mean(
                [self.data['job_types'][r]['ops'][j]['processing_times'][m] for r, j in compat_ops]
            ) if deg_m > 0 else 0.0
            Q_minus = float(sum(self.Q_minus_machine[m].values()))

            gap_terms = []
            fluid_rate_sum = 0.0
            deg_m_f = 0
            for r, j in compat_ops:
                u_val = float(self.u_mrj.get((m, r, j), 0.0))
                fluid_rate_sum += u_val
                if u_val > 1e-9:
                    deg_m_f += 1
                    gap_terms.append(
                        float(self.Q_plus.get((r, j), 0)) -
                        float(snapshot['q_plus_rj'].get((r, j), 0.0))
                    )
            gap_m = np.mean(gap_terms) if deg_m_f > 0 else 0.0

            x_m.append([
                float(self.machines[m]['status']),
                Q_minus / norm_q,
                mean_t / norm_time,
                deg_m / max(1.0, float(len(self.fluid_solver.op_list))),
                gap_m / norm_q,
                fluid_rate_sum,
                deg_m_f / max(1.0, float(len(self.fluid_solver.op_list))),
            ])

        for idx in range(len(self.fluid_solver.op_list)):
            r, j = self._get_op_from_idx(idx)
            compat_m = [int(mm) for mm in self.data['job_types'][r]['ops'][j]['compatible_machines']]

            Q_val = float(self.queues.get((r, j), 0))
            Q_plus_val = float(self.Q_plus.get((r, j), 0))
            deg_o = len(compat_m)
            mean_t_o = np.mean(
                [self.data['job_types'][r]['ops'][j]['processing_times'][mx] for mx in compat_m]
            ) if deg_o > 0 else 0.0
            gap_rj = Q_plus_val - float(snapshot['q_plus_rj'].get((r, j), 0.0))
            gap_ratio = gap_rj / max(1.0, float(self.initial_Q_plus.get((r, j), 0)))
            e_val = float(snapshot['e_rj'].get((r, j), 0.0))
            deg_o_f = sum([1 for mx in compat_m if self.u_mrj.get((mx, r, j), 0.0) > 1e-9])

            x_o.append([
                Q_val / norm_q,
                Q_plus_val / norm_q,
                mean_t_o / norm_time,
                deg_o / max(1.0, float(self.num_machines)),
                gap_rj / norm_q,
                np.clip(gap_ratio, -1.0, 1.0),
                e_val,
                deg_o_f / max(1.0, float(self.num_machines)),
            ])

            for m in compat_m:
                u = float(self.u_mrj.get((m, r, j), 0.0))
                if u > 1e-9:
                    edge_index_om.append([idx, m])
                    t_val = float(self.data['job_types'][r]['ops'][j]['processing_times'][m])
                    Q_minus_mrj = float(self.Q_minus_machine[m].get((r, j), 0))

                    q_plus_mrj_init_discrete = float(snapshot['q_plus_mrj_init'].get((m, r, j), 0.0))
                    Q_plus_mrj_t = max(0.0, q_plus_mrj_init_discrete - Q_minus_mrj)
                    q_plus_mrj_t = float(snapshot['q_plus_mrj'].get((m, r, j), 0.0))
                    gap_mrj = Q_plus_mrj_t - q_plus_mrj_t
                    e_f_mrj = float(snapshot['e_mrj_u'].get((m, r, j), 0.0))

                    edge_feat = [
                        t_val / norm_time,
                        Q_minus_mrj / norm_q,
                        u,
                        q_plus_mrj_t / norm_q,
                        e_f_mrj,
                        gap_mrj / norm_q,
                    ]

                    # 工业实例专用第 7 维：当前机器从前序产品类型切换到候选产品类型的 changeover 时间。
                    if self._use_changeover_feature():
                        changeover_val = self._get_changeover_time(r, j, m)
                        edge_feat.append(changeover_val / norm_time)

                    edge_attr_om.append(edge_feat)

        edge_index_prev, edge_index_next = self._build_op_edges()

        return {
            'x_m': np.array(x_m, dtype=np.float32),
            'x_o': np.array(x_o, dtype=np.float32),
            'edge_index_om': np.array(edge_index_om, dtype=np.int64).T if edge_index_om else np.zeros((2, 0), dtype=np.int64),
            'edge_attr_om': np.array(edge_attr_om, dtype=np.float32) if edge_attr_om else np.zeros((0, self._get_edge_dim()), dtype=np.float32),
            'edge_index_prev': edge_index_prev,
            'edge_index_next': edge_index_next,
            'valid_actions': self.get_valid_actions(),
        }
