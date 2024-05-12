import numpy as np
import os
import torch
from typing import Tuple, Union
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from elegantrl.train import Config, ReplayBuffer


class AgentBase:
    """
    The basic agent of ElegantRL

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.gamma = args.gamma  # discount factor of future rewards
        # self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.num_envs = args.num_envs if args and hasattr(args, 'num_envs') else 1
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        # self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.clip_grad_norm = getattr(args, 'clip_grad_norm', None) or 0.5
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        # self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state
        self.state_value_tau = getattr(args, 'state_value_tau', 0.001)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''network'''
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act
        # self.act = self.act_target = None
        # self.cri = self.cri_target = None
        # self.initialize_networks(net_dims, state_dim, action_dim)

        '''optimizer'''
        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer
        from types import MethodType  # built-in package of Python3
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        """attribute"""
        if self.num_envs == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        self.if_use_per = getattr(args, 'if_use_per', None)  # use PER (Prioritized Experience Replay)
        if self.if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

    # def initialize_networks(self, net_dims: [int], state_dim: int, action_dim: int):
    #     """Dynamically initializes the actor and critic networks based on state_dim"""
    #     act_class = getattr(self, "act_class", None)
    #     cri_class = getattr(self, "cri_class", None)
    #
    #     self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
    #     self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
    #         if cri_class else self.act
    #
    #     # Reinitialize optimizers because network parameters have changed
    #     self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)
    #     self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), self.learning_rate) \
    #         if cri_class else self.act_optimizer
    #
    # def adjust_networks_based_on_state(self, ary_state):
    #     if ary_state is not None:
    #         current_state_dim = ary_state.shape[-1]
    #         if current_state_dim != self.state_dim:
    #             self.state_dim = current_state_dim
    #             self.initialize_networks(self.net_dims, self.state_dim, self.action_dim)  # Assuming this method exists
    #     else:
    #         # Handle the case where ary_state is None, which shouldn't normally happen
    #         print("Warning: Received None state from environment. Check environment reset logic.")

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            num_envs == 1
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        if self.last_state is None:
            # Reset environment and set the initial state
            initial_state, info = env.reset()
            self.last_state = torch.as_tensor(initial_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.adjust_networks_based_on_state(initial_state)

        state = self.last_state  # state.shape == (1, state_dim) for a single env.

        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.rand(1, self.action_dim) * 2 - 1.0 if if_random else get_action(state)
            states[t] = state

            ary_action = action[0].detach().cpu().numpy()
            ary_state, reward, done, info, _ = env.step(ary_action)  # next_state
            if done:
                ary_state, _ = env.reset()  # Ensure a valid state is received
            # ary_state = env.reset() if done else ary_state  # ary_state.shape == (state_dim, )

            self.adjust_networks_based_on_state(ary_state)

            model_input_dim = get_model_input_dim_from_state(ary_state)
            state_dim = ary_state.shape[-1] if ary_state is not None else self.state_dim

            # Use model_input_dim to determine expected shape
            # expected_shape = (1, model_input_dim)
            expected_shape = (1, state_dim)

            state = safe_tensor_conversion(ary_state, expected_shape, device=self.device)
            # state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

            if done:
                break

        self.last_state = state  # state.shape == (1, state_dim) for a single env.

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        if_random: uses random action for warn-up exploration
        return: `(states, actions, rewards, undones)` for off-policy
            states.shape == (horizon_len, num_envs, state_dim)
            actions.shape == (horizon_len, num_envs, action_dim)
            rewards.shape == (horizon_len, num_envs)
            undones.shape == (horizon_len, num_envs)
        """
        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape == (num_envs, state_dim)
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.rand(self.num_envs, self.action_dim) * 2 - 1.0 if if_random \
                else get_action(state).detach()
            states[t] = state  # state.shape == (num_envs, state_dim)

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: Union[ReplayBuffer, tuple]) -> Tuple[float, ...]:
        obj_critic = 0.0  # criterion(q_value, q_label).mean().item()
        obj_actor = 0.0  # q_value.mean().item()
        assert isinstance(buffer, ReplayBuffer) or isinstance(buffer, tuple)
        assert isinstance(self.batch_size, int)
        assert isinstance(self.repeat_times, int)
        assert isinstance(self.reward_scale, float)
        return obj_critic, obj_actor

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            next_as = self.act_target(next_ss)  # next actions
            next_qs = self.cri_target(next_ss, next_as)  # next q values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states, actions)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)

            next_as = self.act_target(next_ss)
            next_qs = self.cri_target(next_ss, next_as)
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states, actions)
        td_errors = self.criterion(q_values, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_action = self.act_target(last_state)
        next_value = self.cri_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_update_amp(self, optimizer: torch.optim, objective: Tensor):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    def update_avg_std_for_normalization(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_target', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            if if_save:
                torch.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, torch.load(file_path, map_location=self.device))


def get_optim_param(optimizer: torch.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list

# Assuming ary_state is already defined and reflects the full state
def get_model_input_dim_from_state(ary_state):
    if isinstance(ary_state, tuple):
        ary_state = ary_state[0]  # Assuming the first element is the relevant state
    if isinstance(ary_state, np.ndarray):
        model_input_dim = ary_state.shape[0] if ary_state.ndim == 1 else ary_state.shape[1]
    else:
        raise ValueError("ary_state format unrecognized.")
    return model_input_dim

def get_expected_state_shape(env, state, model_input_dim):
    print(env)
    if isinstance(state, tuple):
        state_array = state[0]
    else:
        state_array = state

    if isinstance(state_array, np.ndarray):
        if state_array.ndim == 1:
            expected_shape = (1, state_array.shape[0])
        else:
            expected_shape = (1, state_array.shape[1])
    else:
        expected_shape = (1, model_input_dim)

    print(f"Expected Shape at the end: {expected_shape}")
    return expected_shape

def safe_tensor_conversion(ary_state, expected_shape, device):
    # Handle different data types and structures of ary_state
    if isinstance(ary_state, tuple):
        # Assuming the first element of the tuple is the array and the second element is a dict
        ary_state, _ = ary_state  # Ignore the dict part and only take the array part
        if isinstance(ary_state, np.ndarray) and ary_state.shape != expected_shape:
            # print(f"Warning: ary_state ndarray shape mismatch. Expected {expected_shape}, got {ary_state.shape}. Setting to default.")
            ary_state = np.zeros(expected_shape, dtype=np.float32)

    elif isinstance(ary_state, np.ndarray):
        if ary_state.shape != expected_shape:
            # print(f"Warning: ary_state ndarray shape mismatch. Expected {expected_shape}, got {ary_state.shape}. Setting to default.")
            ary_state = np.zeros(expected_shape, dtype=np.float32)
    else:
        # print(f"Warning: ary_state type {type(ary_state)} unrecognised. Setting to default.")
        ary_state = np.zeros(expected_shape, dtype=np.float32)

    # Proceed with the conversion now that ary_state is guaranteed to be correct
    return torch.as_tensor(ary_state, dtype=torch.float32, device=device)
