import copy
import numpy as np

try:
    import torch
except Exception:
    torch = None

from env import FJSPEnv


def _zero_columns(x, cols):
    if x is None:
        return x

    if torch is not None and isinstance(x, torch.Tensor):
        y = x.clone()
        if y.ndim == 2 and y.shape[1] > 0:
            valid_cols = [c for c in cols if 0 <= c < y.shape[1]]
            if len(valid_cols) > 0:
                y[:, valid_cols] = 0.0
        return y

    if isinstance(x, np.ndarray):
        y = x.copy()
        if y.ndim == 2 and y.shape[1] > 0:
            valid_cols = [c for c in cols if 0 <= c < y.shape[1]]
            if len(valid_cols) > 0:
                y[:, valid_cols] = 0.0
        return y

    try:
        y = copy.deepcopy(x)
        if len(y) > 0 and isinstance(y[0], (list, tuple)):
            width = len(y[0])
            valid_cols = [c for c in cols if 0 <= c < width]
            for i in range(len(y)):
                row = list(y[i])
                for c in valid_cols:
                    row[c] = 0.0
                y[i] = row
        return y
    except Exception:
        return x


class AblationFJSPEnv(FJSPEnv):
    """
    在不改原 env.py 的前提下，做最小侵入式消融包装：
    1) use_fluid_mask=False:
       不再用 u_mrj > 0 硬掩码动作
    2) use_fluid_state=False:
       状态中流体相关特征置零（维度不变）
    3) use_fluid_reward_scale=False:
       奖励去掉流体分母（通过 super.step 的 reward 乘回分母实现）

    工业实例 changeover 扩展后：
    - edge_attr_om 原 6 列顺序不变；
    - 第 7 列 changeover_time 不是流体特征，消融 use_fluid_state=False 时不置零。
    """

    OP_FLUID_COLS = [4, 5, 6, 7]
    M_FLUID_COLS = [3, 4, 5, 6]
    EDGE_FLUID_COLS = [2, 3, 4, 5]

    def __init__(self, data, ablation_cfg=None):
        self.ablation_cfg = ablation_cfg or {}
        super(AblationFJSPEnv, self).__init__(data)

    def _use_fluid_mask(self):
        return bool(self.ablation_cfg.get('use_fluid_mask', True))

    def _use_fluid_state(self):
        return bool(self.ablation_cfg.get('use_fluid_state', True))

    def _use_fluid_reward_scale(self):
        return bool(self.ablation_cfg.get('use_fluid_reward_scale', True))

    # -----------------------------------------------------
    # 变体 1：取消 u_mrj = 0 的动作硬掩码
    # -----------------------------------------------------
    def get_valid_actions(self):
        if self._use_fluid_mask():
            return super(AblationFJSPEnv, self).get_valid_actions()

        actions = []
        for r_idx, job_type in enumerate(self.data['job_types']):
            for j, op in enumerate(job_type['ops']):
                if self.queues.get((r_idx, j), 0) <= 0:
                    continue

                compat_m = op['compatible_machines']
                for m in compat_m:
                    if self.machines[m]['status'] == 0:
                        actions.append((self._get_op_idx(r_idx, j), m))
        return actions

    # -----------------------------------------------------
    # 变体 2：去除状态中的流体特征（置零）
    # -----------------------------------------------------
    def _get_state(self):
        state = super(AblationFJSPEnv, self)._get_state()

        if not self._use_fluid_mask():
            state['valid_actions'] = self.get_valid_actions()

        if self._use_fluid_state():
            return state

        # 兼容旧命名 op_feat/m_feat/edge_feat 和当前 env.py 返回的
        # x_o/x_m/edge_attr_om。
        for key in ['op_feat', 'x_o']:
            if key in state:
                state[key] = _zero_columns(state[key], self.OP_FLUID_COLS)

        for key in ['m_feat', 'x_m']:
            if key in state:
                state[key] = _zero_columns(state[key], self.M_FLUID_COLS)

        for key in ['edge_feat', 'edge_attr_om']:
            if key in state:
                state[key] = _zero_columns(state[key], self.EDGE_FLUID_COLS)

        return state

    # -----------------------------------------------------
    # 变体 3：去除奖励中的流体分母
    # -----------------------------------------------------
    def _get_fluid_reward_denominator(self):
        candidate_names = [
            'fluid_Cmax',
            'fluid_cmax',
            'fluid_makespan',
            'Cmax_fluid',
            'C_fluid',
            'fluid_C'
        ]

        for name in candidate_names:
            if hasattr(self, name):
                try:
                    v = float(getattr(self, name))
                    if v > 1e-9:
                        return v
                except Exception:
                    pass

        if hasattr(self, 'fluid_solver'):
            solver = getattr(self, 'fluid_solver')
            for name in ['Cmax', 'cmax', 'makespan', 'fluid_Cmax', 'fluid_cmax']:
                if hasattr(solver, name):
                    try:
                        v = float(getattr(solver, name))
                        if v > 1e-9:
                            return v
                    except Exception:
                        pass

        return 1.0

    def step(self, action):
        next_state, reward, done, info = super(AblationFJSPEnv, self).step(action)

        if self._use_fluid_reward_scale():
            return next_state, reward, done, info

        denom = self._get_fluid_reward_denominator()
        reward = float(reward) * float(denom)
        return next_state, reward, done, info
