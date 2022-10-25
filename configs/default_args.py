from params_proto import ParamsProto, Proto

class Args(ParamsProto):

    # len for pos is 2, if you want to repeat pos and vol, change this to 4
    repeat_len = 2
    include_goal_in_state = False
    pin_goal=True
    shorten_ratio = 1.0

    action_dim = 2
    action_weight = 1
    batch_size = 32

    maze_dataset = Proto('/data/maze2d/maze2d-open-v0.hdf5',env="MAZE_DATASET",
                     help="Path to maze dataset")

    bucket = None
    clip_denoised = True
    commit = None
    config = 'config.maze2d'
    dataset = 'maze2d-open-v0'
    env_name = 'maze2d-open-v0'
    device = 'cuda'
    diffusion = 'models.GaussianDiffusion'
    dim_mults = (1,4,8)
    ema_decay = 0.995
    exp_name = 'diffusion/H384_T256'
    gradient_accumulate_every = 2
    horizon = 384
    short_horizon = 384
    learning_rate = 0.0002
    loader = 'datasets.GoalDataset'
    logbase = 'logs'
    loss_discount = 1
    loss_type = 'l2'
    loss_weights = None
    max_path_length = 40000
    model = 'models.TemporalUnet'
    n_diffusion_steps = 256
    n_reference = 50
    n_samples = 5
    n_saves = 50
    n_steps_per_epoch = 10000
    n_train_steps = 1000000.0
    normalizer = 'LimitsNormalizer'
    observation_dim = 4
    predict_epsilon = False
    prefix = 'diffusion/'
    preprocess_fns = ['maze2d_set_terminals']
    renderer = 'utils.Maze2dRenderer'
    sample_freq = 5000
    save_freq = 100000
    log_freq = 100
    save_parallel = False
    snapshot_root = '/models'
    termination_penalty = None
    use_padding = False