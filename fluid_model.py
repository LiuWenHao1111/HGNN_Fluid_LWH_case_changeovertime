import warnings
import numpy as np
from scipy.optimize import linprog

warnings.filterwarnings('ignore')


class FluidModel(object):
    """
    Paper-aligned fluid model.

    对应论文 4.2 / 表3 / 式(1.2)~(1.9) 的核心量：
        e_mrj   = 1 / t_mrj
        u_mrj   = machine m 对工序类型 o_rj 的单位时间分配比例
        e_rj    = sum_m (e_mrj * u_mrj)
        c_rj^f  = q_rj^+(0) / e_rj

        q_rj^+(t)   = max(0, q_rj^+(0) - e_rj * t)
        q_mrj^+(0)  = q_rj^+(0) * (e_mrj * u_mrj) / e_rj
        q_mrj^+(t)  = max(0, q_mrj^+(0) - e_mrj * u_mrj * t)

    说明：
    1) 论文把流体模型写成线性模型，但没有给出一个完整的“时间索引离散 LP”实现细节。
    2) 为了保持你当前 env.py / hgnn.py / mappo.py / main.py 接口完全不变，
       这里采用“静态 LP 求 u_mrj + 闭式恢复连续状态”的实现方式。
    3) 这比你当前版本更严格地显式对齐了：
       - stage 初始在制数 q_rj(0)
       - 工序链速率约束
       - q_rj(t), q_rj^+(t), q_mrj^+(0), q_mrj^+(t), c_rj^f
    """

    def __init__(self, data):
        self.data = data
        self.num_machines = int(data['num_machines'])
        self.job_types = data['job_types']

        self.op_list = []
        self.op_to_idx = {}
        self.compatible_pairs = []
        self.proc_time = {}
        self.base_rate = {}

        idx = 0
        for r_idx, jt in enumerate(self.job_types):
            for op in jt['ops']:
                j = int(op['op_id'])
                self.op_list.append((r_idx, j))
                self.op_to_idx[(r_idx, j)] = idx
                idx += 1

                for m in op['compatible_machines']:
                    m = int(m)
                    t = float(op['processing_times'][m])
                    self.compatible_pairs.append((m, r_idx, j))
                    self.proc_time[(m, r_idx, j)] = t
                    self.base_rate[(m, r_idx, j)] = 1.0 / t

        self.num_total_ops = idx

    def _build_stage_initial_q(self, initial_Q_plus):
        """
        论文中 q_rj(0) 表示“处在工序 o_rj 阶段的工件总数”。
        对于当前问题：t=0 时只有首工序阶段有工件，其余阶段为 0。
        """
        q_stage_0 = {}
        for (r, j) in self.op_list:
            if j == 0:
                # 首工序在制数 = 该类型批量
                q_stage_0[(r, j)] = float(initial_Q_plus.get((r, j), 0.0))
            else:
                q_stage_0[(r, j)] = 0.0
        return q_stage_0

    def solve(self, initial_Q_plus):
        """
        Solve LP for:
            C_max^f
            x_mrj = C_max^f * u_mrj

        LP 变量:
            z[0] = C_max^f
            z[col(m,r,j)] = x_mrj

        约束对应论文含义：
            1) 机器容量约束
            2) 各工序未加工总量完成时间约束
            3) 工序链速率约束（对应 q_rj(0)=0 时，下游流速不超过上游）
        """
        pair_to_col = {}
        for col_idx, pair in enumerate(self.compatible_pairs, start=1):
            pair_to_col[pair] = col_idx

        num_vars = 1 + len(self.compatible_pairs)
        c = np.zeros(num_vars, dtype=np.float64)
        c[0] = 1.0  # min C_max^f

        A_ub = []
        b_ub = []

        # ---------------------------------------------------------
        # 1) 机器容量约束：
        #    sum_{o_rj in O_m^RJ} u_mrj <= 1
        # -> sum x_mrj - C_max^f <= 0
        # ---------------------------------------------------------
        for m in range(self.num_machines):
            row = np.zeros(num_vars, dtype=np.float64)
            row[0] = -1.0
            for (mm, r, j) in self.compatible_pairs:
                if mm == m:
                    row[pair_to_col[(mm, r, j)]] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)

        # ---------------------------------------------------------
        # 2) 完工时间约束：
        #    c_rj^f = q_rj^+(0) / e_rj
        #    且 C_max^f >= c_rj^f
        # -> q_rj^+(0) <= C_max^f * e_rj = sum_m e_mrj * x_mrj
        # ---------------------------------------------------------
        for (r, j) in self.op_list:
            q_plus_0 = float(initial_Q_plus.get((r, j), 0.0))
            row = np.zeros(num_vars, dtype=np.float64)
            compat = [int(m) for m in self.job_types[r]['ops'][j]['compatible_machines']]
            for m in compat:
                row[pair_to_col[(m, r, j)]] = -self.base_rate[(m, r, j)]
            A_ub.append(row)
            b_ub.append(-q_plus_0)

        # ---------------------------------------------------------
        # 3) 工序链速率约束（对应论文中 q_rj(0)=0 时的平衡关系）
        #    e_rj <= e_r,j-1,  j >= 1
        # -> sum_m e_mrj x_mrj - sum_m e_m,r,j-1 x_m,r,j-1 <= 0
        # ---------------------------------------------------------
        q_stage_0 = self._build_stage_initial_q(initial_Q_plus)
        for r, jt in enumerate(self.job_types):
            for j in range(1, len(jt['ops'])):
                if q_stage_0[(r, j)] <= 1e-12:
                    row = np.zeros(num_vars, dtype=np.float64)

                    for m in jt['ops'][j]['compatible_machines']:
                        m = int(m)
                        row[pair_to_col[(m, r, j)]] += self.base_rate[(m, r, j)]

                    for m in jt['ops'][j - 1]['compatible_machines']:
                        m = int(m)
                        row[pair_to_col[(m, r, j - 1)]] -= self.base_rate[(m, r, j - 1)]

                    A_ub.append(row)
                    b_ub.append(0.0)

        # 变量边界：C_max^f > 0, x_mrj >= 0
        bounds = [(1e-8, None)] + [(0.0, None)] * len(self.compatible_pairs)

        try:
            res = linprog(
                c,
                A_ub=np.array(A_ub, dtype=np.float64),
                b_ub=np.array(b_ub, dtype=np.float64),
                bounds=bounds,
                method='highs'
            )
        except Exception:
            res = linprog(
                c,
                A_ub=np.array(A_ub, dtype=np.float64),
                b_ub=np.array(b_ub, dtype=np.float64),
                bounds=bounds,
                method='interior-point'
            )

        u_result = {}
        for pair in self.compatible_pairs:
            u_result[pair] = 0.0

        C_max_f = 1.0

        if res is not None and res.success and res.x[0] > 1e-8:
            C_max_f = float(res.x[0])
            for pair in self.compatible_pairs:
                x_val = float(res.x[pair_to_col[pair]])
                u_result[pair] = max(0.0, x_val / C_max_f)
        else:
            # fallback: 对每道工序按基础速率比例分配
            for (r, j) in self.op_list:
                compat = [int(m) for m in self.job_types[r]['ops'][j]['compatible_machines']]
                denom = sum([self.base_rate[(m, r, j)] for m in compat])
                for m in compat:
                    u_result[(m, r, j)] = self.base_rate[(m, r, j)] / max(denom, 1e-8)

            snapshot = self.get_fluid_state(u_result, initial_Q_plus, 0.0)
            if len(snapshot['c_rj']) > 0:
                C_max_f = max(snapshot['c_rj'].values())
            else:
                C_max_f = 1.0

        # 数值清理：保证每台机器总分配比例不超过 1
        for m in range(self.num_machines):
            total = 0.0
            for (r, j) in self.op_list:
                if (m, r, j) in u_result:
                    total += u_result[(m, r, j)]
            if total > 1.0 + 1e-8:
                scale = 1.0 / total
                for (r, j) in self.op_list:
                    if (m, r, j) in u_result:
                        u_result[(m, r, j)] *= scale

        return u_result, float(max(C_max_f, 1.0))

    def get_fluid_state(self, u_mrj, initial_Q_plus, time_t):
        """
        Recover paper-aligned continuous quantities:
            q_rj(t)           : 当前处在工序阶段 rj 的流体量
            q_rj^+(t)         : 尚未完成工序 rj 的流体总量
            q_mrj^+(0)
            q_mrj^+(t)
            e_rj
            e_mrj * u_mrj
            c_rj^f
        """
        q_stage_rj = {}
        q_plus_rj = {}
        q_plus_mrj = {}
        q_plus_mrj_init = {}
        e_rj = {}
        e_mrj_u = {}
        c_rj = {}

        t = float(time_t)
        q_stage_0 = self._build_stage_initial_q(initial_Q_plus)

        # 先求每道工序的总流速 e_rj
        for (r, j) in self.op_list:
            compat = [int(m) for m in self.job_types[r]['ops'][j]['compatible_machines']]
            total_rate = 0.0
            for m in compat:
                pair = (m, r, j)
                eff_rate = self.base_rate[pair] * float(u_mrj.get(pair, 0.0))
                e_mrj_u[pair] = eff_rate
                total_rate += eff_rate
            e_rj[(r, j)] = total_rate

        # 再按论文公式恢复 q^+, q_m^+, c^f
        for (r, j) in self.op_list:
            q_plus_0 = float(initial_Q_plus.get((r, j), 0.0))
            rate_total = float(e_rj.get((r, j), 0.0))

            # q_rj^+(t)
            q_plus_rj[(r, j)] = max(0.0, q_plus_0 - rate_total * t)

            # c_rj^f
            c_rj[(r, j)] = q_plus_0 / max(rate_total, 1e-8)

            # q_rj(t) —— 处在当前工序阶段的流体量
            if j == 0:
                q_stage_rj[(r, j)] = max(0.0, q_stage_0[(r, j)] - rate_total * t)
            else:
                prev_rate = float(e_rj.get((r, j - 1), 0.0))
                q_stage_rj[(r, j)] = max(0.0, q_stage_0[(r, j)] + (prev_rate - rate_total) * t)

            # q_mrj^+(0), q_mrj^+(t)
            compat = [int(m) for m in self.job_types[r]['ops'][j]['compatible_machines']]
            for m in compat:
                pair = (m, r, j)
                share = e_mrj_u[pair] / max(rate_total, 1e-8)
                q0_m = q_plus_0 * share
                q_plus_mrj_init[pair] = q0_m
                q_plus_mrj[pair] = max(0.0, q0_m - e_mrj_u[pair] * t)

        return {
            'q_rj': q_stage_rj,              # 新增，不影响现有其他文件
            'q_plus_rj': q_plus_rj,
            'q_plus_mrj': q_plus_mrj,
            'q_plus_mrj_init': q_plus_mrj_init,
            'e_rj': e_rj,
            'e_mrj_u': e_mrj_u,
            'c_rj': c_rj,
        }