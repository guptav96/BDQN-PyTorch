#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *

# torch.set_printoptions(profile="full")
# np.set_printoptions(threshold=1000000)

def bdqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    
    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.0025/4, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: BDQNNet(NatureConvBody(in_channels=config.history_length))
    config.batch_size = 32
    config.discount = 0.99
    config.history_length = 4
    config.max_steps = int(5e7)
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length,
    )
    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.6
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 20000
    config.sgd_update_frequency = 4
    config.bdqn_learn_frequency = 10000
    config.thompson_sampling_frequency = 1000
    config.prior_var = 0.001
    config.noise_var = 2
    config.var_k = 0.001
#     config.gradient_clip = 1
    config.double_q = True
    config.async_actor = False
    run_steps(BDQNAgent(config))

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(3)

    game = 'BankHeistNoFrameskip-v4'
    bdqn_pixel(game=game, n_step=1, replay_cls=PrioritizedReplay, async_replay=True, run=0, remark='exp1')
