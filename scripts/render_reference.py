def main(__init_wandb=True):
    import helper.utils as utils
    from configs.default_args import Args
    import helper.datasets as datasets
    import wandb
    import torch
    from helper.utils.arrays import to_np
    import numpy as np

    # Args._update(kwargs)

    if __init_wandb:
        wandb.login()
        wandb.init(project=f"SharedAutoMaze2D", config=vars(Args), name=f"down-open", reinit=True)

    def get_dataset(Args):
        params = {
            'env': Args.env_name,
            'horizon': Args.horizon,
            'normalizer': Args.normalizer,
            'preprocess_fns': Args.preprocess_fns,
            'use_padding': Args.use_padding,
            'max_path_length': Args.max_path_length
        }

        if Args.pin_goal:
            dataset = datasets.GoalDataset(**params)
        else:
            dataset = datasets.SequenceDataset(**params)

        test_dataset = datasets.GoalDataset(**params)

        return dataset, test_dataset

    def cycle(dl):
        while True:
            for data in dl:
                yield data

    def render_reference(dataset, batch_size=20):
        '''
            renders training points
        '''

        renderer = utils.Maze2dRenderer(env=Args.env_name)

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, dataset.action_dim:]
        observations = dataset.normalizer.unnormalize(normed_observations, 'observations')

        # Get maze info
        # if Args.use_switch_umaze:
        #     from maze_exp.launcher.takuma.mazeid_mapping import id2mazename
        #     mazeids = observations[:, 0, -1]
        #     maze_names = [id2mazename[_id] for _id in mazeids]
        #     updated_background = [self.envs[env_name].maze_arr == 10 for env_name in maze_names]
        # else:
        #     maze_names = None
        #     updated_background = None

        # Remove extra dimensions
        # from maze_exp.launcher.takuma.mazeid_mapping import get_extra_dims
        # extra_dim = get_extra_dims()
        # observations = observations[:, :, :-extra_dim]

        # NOTE: env_name is used only to decide the boundary.
        savepath = None
        images = renderer.composite(savepath, observations, ncol=10, env_name=Args.env_name)
        wandb.log({f"demonstration": wandb.Image(np.array(images), caption=f'demonstration')})

    dataset, test_dataset = get_dataset(Args)

    render_reference(dataset)


if __name__ == '__main__':
    # NOTE: You will not execute this if you're using Jaynes!!
    import os
    import argparse
    import wandb
    from configs.default_args import Args

    # parser = argparse.ArgumentParser()
    # parser.add_argument("sweep_file", type=str,
    #                     help="sweep file")
    # parser.add_argument("-l", "--line-number",
    #                     type=int, help="line number of the sweep-file")
    # parser.add_argument("--cvd", default=0,
    #                     type=int,
    #                     help="setup CUDA_VISIBLE_DEVICES given line number. Specify the maximum value for CUDA_VISIBLE_DEVICES")
    # args = parser.parse_args()
    #
    # # Obtain kwargs from Sweep
    # from params_proto.hyper import Sweep
    # from maze_exp import RUN

    # sweep = Sweep(RUN, Args).load(args.sweep_file)
    # kwargs = list(sweep)[args.line_number]
    #
    # if args.cvd > 0:
    #     import os
    #
    #     cvd = str(args.line_number % args.cvd)
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cvd
    #     print(f'setting CUDA_VISIBLE_DEVICES: {cvd}')

    wandb.login()  # NOTE: You need to set envvar WANDB_API_KEY
    wandb.init(
        # Set the project where this run will be logged
        project='shared_maze2d',
        # Track hyperparameters and run metadata
        config=vars(Args),
        # resume=True,
        # id=wandb_runid
    )

    # Oftentimes crash logs are not viewable on wandb webapp. This uploads the slurm log.
    # wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out', policy='end')
    # wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out.log', policy='end')
    # wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.error.log', policy='end')

    main(__init_wandb=False)
    wandb.finish()
