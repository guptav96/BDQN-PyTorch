from bdqn import *

def bdqn_experiment(**kwargs):
    
    def cbk_record_episode(episode_id):
        return episode_id % 10 == 0 and agent.eval_mode == True
    
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, config.remark, cbk_record_episode)
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
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 20000
    config.sgd_update_frequency = 4
    config.bdqn_learn_frequency = 100000
    config.thompson_sampling_frequency = 1000
    config.prior_var = 0.001
    config.noise_var = 1
    config.var_k = 0.001
    config.double_q = True
    config.async_actor = False
    agent = BDQNAgent(config)
    run_steps(agent)

if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    select_device(1)

    game = 'BattleZoneNoFrameskip-v4'
    bdqn_experiment(game=game, n_step=3, replay_cls=PrioritizedReplay, async_replay=True, run=0, remark='test10')
