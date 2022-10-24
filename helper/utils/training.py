import os
import copy
import numpy as np
import torch
import einops
import pdb
import wandb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
import helper.datasets as datasets
from configs.default_args import Args

# from ml_logger import logger

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        test_dataset,
        renderer,
        results_folder,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.test_dataset = test_dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder+f'/include-{Args.include_goal_in_state}/pingoal-{Args.pin_goal}/horizon-{Args.short_horizon}'
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

        self.env=datasets.load_environment(Args.env_name)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
            os.makedirs(os.path.join(self.logdir,"param"))
            os.makedirs(os.path.join(self.logdir, "ref"))
            os.makedirs(os.path.join(self.logdir, "eval/pre"))
            os.makedirs(os.path.join(self.logdir, "eval/obs"))
            os.makedirs(os.path.join(self.logdir, "eval/act"))
            os.makedirs(os.path.join(self.logdir, "eval/ref"))

        wandb.define_metric("train-loss", step_metric="epoch", summary="min")
        wandb.define_metric("act-reward", step_metric="epoch", summary="max")
        wandb.define_metric("obs-reward", step_metric="epoch", summary="max")
        wandb.define_metric("ref-reward", step_metric="epoch")

        wandb.define_metric("obs.vs.ref", step_metric="epoch", summary="max")
        wandb.define_metric("act.vs.ref", step_metric="epoch", summary="max")

        wandb.define_metric("act-reward-var", step_metric="epoch", summary="max")
        wandb.define_metric("obs-reward-var", step_metric="epoch", summary="max")

        wandb.define_metric("obs-reward-25", step_metric="epoch")
        wandb.define_metric("act-reward-25", step_metric="epoch")



    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        loss_history=[]

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)

                if Args.include_goal_in_state:
                    curr_traj = batch.trajectories
                    new_traj = torch.zeros((curr_traj.shape[0], curr_traj.shape[1], curr_traj.shape[2] + Args.repeat_len))
                    for index in range(curr_traj.shape[0]):
                        goal_pos = curr_traj[index, -1, Args.action_dim:Args.action_dim+Args.repeat_len].repeat(curr_traj.shape[1], 1)
                        new_traj[index] = torch.hstack((curr_traj[index], goal_pos))
                    '''
                    batch.trajectories=new_traj.to(Args.device)
                    --> raise error AttributeError: can't set attribute
                    reason: in sequence dataset, create dict ny namedtuple, an immutable object
                    solution: https://blog.finxter.com/solved-attributeerror-cant-set-attribute-in-python/
                    '''
                    cons=batch.conditions.copy()

                    for key in cons.keys():
                        con=cons[key]
                        con=torch.hstack((con,con[:,:Args.repeat_len]))
                        cons[key]=con

                    batch=batch._replace(trajectories=new_traj,conditions=cons)

                batch = batch_to_device(batch)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every

                loss_history.append(loss.cpu().item())

                # logger.store_metrics(loss=loss.cpu().item())

                if self.step % self.log_freq == 0:
                    log_dict={
                        "train-loss": np.array(loss_history).mean(),
                        "epoch": self.step,
                    }
                    wandb.log(log_dict)
                    loss_history = []

                log_dict={
                    "train-loss": loss.cpu().item(),
                    "epoch": self.step,
                }
                wandb.log(log_dict)

                loss.backward()

            # logger.log_metrics_summary(key_values={"step": self.step})
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0 or self.step==Args.n_train_steps-1:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }

        curr=(self.step-2*self.save_freq) // self.label_freq * self.label_freq
        if os.path.exists(os.path.join(self.logdir, f'param/state_{curr}.pt')):
            os.remove(os.path.join(self.logdir, f'param/state_{curr}.pt'))

        savepath = os.path.join(self.logdir, f'param/state_{epoch}.pt')
        torch.save(data, savepath)

        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'param/state_{epoch}.pt')
        print("parameters path",loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        if Args.include_goal_in_state:
            normed_observations = trajectories[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]
        else:
            normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        # self.renderer.composite(savepath, observations)
        savepath=os.path.join(self.logdir,"ref/demonstration.png")
        images=self.renderer.composite(savepath, observations)
        Image=wandb.Image(np.array(images), caption=f'demonstration')
        wandb.log({f"demonstration": Image})

        # logger.print("image type:",type(images))
        # logger.print("what is image:",images)
        # logger.print("image shape",np.array(images).shape)
        # logger.save_image(np.array(images),'ref/ref.png')



    def render_samples(self, batch_size=5, n_samples=5):
        '''
            renders samples from (ema) diffusion model
            # batch size decide how many (start,goal) pairs we have
            # n samples decide how many trails we test for each pair
        '''
        exe_act_score = []
        exe_obs_score = []
        reference_score = []
        images1 = []
        images2 = []
        ref_img = []
        pre_image = []
        for i in range(batch_size):
            print("start sampling")

            batch = self.test_dataset[i*200]
            trajectories = np.array(batch.trajectories)
            trajectories = np.expand_dims(trajectories,axis=0)

            for key in batch.conditions.keys():
                batch.conditions[key] = torch.tensor(np.expand_dims(batch.conditions[key], axis=0))

            if Args.include_goal_in_state:
                cons = batch.conditions.copy()
                for key in cons.keys():
                    con = cons[key]
                    # copy current condition and repeat if need
                    con = torch.hstack((con, con[:, :Args.repeat_len]))
                    cons[key] = con

                batch = batch._replace(conditions=cons)

            conditions = to_device(batch.conditions, 'cuda:0')

            start_info = np.array(conditions[0].cpu())[:, None]
            if Args.include_goal_in_state:
                start_info = start_info[...,:-Args.repeat_len]
            start_info = self.dataset.normalizer.unnormalize(start_info, 'observations')
            x = start_info[0][0][0]
            y = start_info[0][0][1]
            xv = start_info[0][0][2]
            yv = start_info[0][0][3]
            start_pos = np.array([x, y])
            start_vel = np.array([xv, yv])

            targets_info = np.array(conditions[383].cpu())[:, None]
            if Args.include_goal_in_state:
                targets_info = targets_info[...,:-Args.repeat_len]
            targets_info = self.dataset.normalizer.unnormalize(targets_info,'observations')
            x = targets_info[0][0][0]
            y = targets_info[0][0][1]
            xv = targets_info[0][0][2]
            yv = targets_info[0][0][3]
            target = np.array([x, y])
            all_tar = np.array([x, y, xv, yv])

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions,horizon=Args.horizon)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            if Args.include_goal_in_state:
                normed_observations = samples[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]
            else:
                normed_observations = samples[:, :, self.dataset.action_dim:]
            normed_actions = samples[:, :, :self.dataset.action_dim]

            # [ 1 x 1 x observation_dim ]
            if Args.include_goal_in_state:
                normed_conditions = to_np(batch.conditions[0])[:, None]
                normed_conditions = normed_conditions[..., :self.dataset.observation_dim]
                normed_reference_observations = trajectories[:, :,
                                                self.dataset.action_dim:self.dataset.action_dim + self.dataset.observation_dim]
                normed_reference_actions = trajectories[:, :,
                                           :self.dataset.action_dim]
            else:
                normed_conditions = to_np(batch.conditions[0])[:, None]
                normed_reference_observations = trajectories[:, :, self.dataset.action_dim:]
                normed_reference_actions = trajectories[:, :,
                                           :self.dataset.action_dim]

            reference_observations = self.dataset.normalizer.unnormalize(normed_reference_observations, 'observations')
            reference_actions = self.dataset.normalizer.unnormalize(normed_reference_actions, 'actions')

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            for o in observations:
                pre_image.append(o)

            ## execute actions
            ## [ n_samples x horizon x action_dim ]
            actions_set=self.dataset.normalizer.unnormalize(normed_actions, 'actions')

            '''
                Below to get reference image
            '''

            # execute the reference actions to see the reward
            self.env.reset()
            # set initial position
            self.env.set_state(start_pos, start_vel)
            obs = self.env._get_obs()
            # set target position
            self.env.set_target(target)
            total_reward = 0

            exe_obs = [obs.copy()]
            reference_actions = reference_actions[0]
            for t in range(reference_actions.shape[0]):
                action = reference_actions[t]
                next_obs, reward, terminal, _ = self.env.step(action)
                total_reward += reward
                # score = total_reward
                score = self.env.get_normalized_score(total_reward)
                exe_obs.append(next_obs.copy())
                if t == reference_actions.shape[0] - 1 or terminal:
                    # manully set the final state to be target state
                    exe_obs.append(all_tar)
                    exe_obs.append(all_tar)
                    # save image name processing
                    ref_img.append(exe_obs)
            reference_score.append(score)

            for n in range(n_samples):
                # execute the exp with actions
                ## [ 1 x horizon x action_dim ]
                actions = actions_set[n]
                # reset environment
                self.env.reset()
                # set initial position
                self.env.set_state(start_pos, start_vel)
                obs = self.env._get_obs()
                # set target position
                self.env.set_target(target)
                total_reward = 0

                exe_obs = [obs.copy()]
                for t in range(actions.shape[0]):
                    action = actions[t]
                    next_obs, reward, terminal, _ = self.env.step(action)
                    total_reward += reward
                    score = self.env.get_normalized_score(total_reward)
                    exe_obs.append(next_obs.copy())
                    if t == actions.shape[0] - 1 or terminal:
                        # manully set the final state to be target state
                        exe_obs.append(all_tar)
                        # save image name processing
                        images1.append(exe_obs)
                exe_act_score.append(score)

                # execute the exp with observation changes
                self.env.reset()
                # set initial position
                self.env.set_state(start_pos, start_vel)
                obs = self.env._get_obs()
                # set target position
                self.env.set_target(target)
                total_reward = 0

                exe_obs = [obs.copy()]
                target = self.env._target
                cond = {
                    self.model.horizon - 1: np.array([*target, 0, 0]),
                }
                for t in range(actions.shape[0]):
                    state = self.env.state_vector().copy()

                    if t == 0:
                        cond[0] = obs.copy()
                        sequence = observations[n]
                    if t < len(sequence) - 1:
                        next_waypoint = sequence[t + 1]
                    else:
                        next_waypoint = sequence[-1].copy()
                        next_waypoint[2:] = 0

                    ## can use actions or define a simple controller based on state predictions
                    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                    next_obs, reward, terminal, _ = self.env.step(action)
                    total_reward += reward
                    score = self.env.get_normalized_score(total_reward)
                    exe_obs.append(next_obs.copy())
                    if t == actions.shape[0] - 1 or terminal:
                        exe_obs.append(all_tar)
                        images2.append(exe_obs)
                exe_obs_score.append(score)

        stp = int(self.step / self.sample_freq)
        index = (str(stp) + 'k').zfill(5)

        savepath = os.path.join(self.logdir, f'eval/act/{index}.png')
        con_image1 = self.renderer.composite(savepath, np.array(images1), ncol=n_samples)
        wbimg1=wandb.Image(np.array(con_image1),caption=f'act-{index}')
        wandb.log({f"exe predict act": wbimg1})

        savepath = os.path.join(self.logdir, f'eval/obs/{index}.png')
        con_image2 = self.renderer.composite(savepath, np.array(images2), ncol=n_samples)
        wbimg2 = wandb.Image(np.array(con_image2), caption=f'obs-{index}')
        wandb.log({f"exe predict obs": wbimg2})

        savepath = os.path.join(self.logdir, f'eval/pre/{index}.png')
        images = self.renderer.composite(savepath, np.array(pre_image), ncol=n_samples)
        wbimgpre = wandb.Image(np.array(images), caption=f'pre-{index}')
        wandb.log({f"predict obs": wbimgpre})

        savepath = os.path.join(self.logdir, f'eval/ref/{index}.png')
        con_image2 = self.renderer.composite(savepath, np.array(ref_img), ncol=1)
        wbimg2 = wandb.Image(np.array(con_image2), caption=f'ref-{index}')
        wandb.log({f"demonstration": wbimg2})

        log_dict = {
                    "obs-reward-25": wandb.Histogram(exe_obs_score),
                    "act-reward-25": wandb.Histogram(exe_act_score),
                    "act-reward": np.array(exe_act_score).mean(),
                    "obs-reward": np.array(exe_obs_score).mean(),
                    "ref-reward": np.array(reference_score).mean(),
                    "obs.vs.ref": np.array(exe_obs_score).mean()/np.array(reference_score).mean(),
                    "act.vs.ref": np.array(exe_act_score).mean()/np.array(reference_score).mean(),
                    "obs-reward-var": np.array(exe_obs_score).var(),
                    "act-reward-var": np.array(exe_obs_score).var(),
                    "epoch": self.step,
                }
        wandb.log(log_dict)


    def render_samples_with_ref_reward(self, batch_size=5, n_samples=5):
        '''
            renders samples from (ema) diffusion model
            # batch size decide how many (start,goal) pairs we have
            # n samples decide how many trails we test for each pair
        '''

        # wandb.define_metric(f"reward-{n_samples}", step_metric="case")
        wandb.define_metric("reference-reward", step_metric="case")
        wandb.define_metric("obs.vs.ref", step_metric="case")
        wandb.define_metric("act.vs.ref", step_metric="case")
        wandb.define_metric("act-reward", step_metric="case")
        wandb.define_metric("obs-reward", step_metric="case")
        # wandb.define_metric("act-reward-var", step_metric="case")
        # wandb.define_metric("obs-reward-var", step_metric="case")

        total_act_var=[]
        total_act_mean=[]
        total_obs_var=[]
        total_obs_mean=[]
        total_ref_mean=[]

        for i in range(batch_size):
            exe_act_score = []
            exe_obs_score = []
            print("start sampling")
            batch = self.test_dataset[i*20]
            trajectories = np.array(batch.trajectories)
            trajectories = np.expand_dims(trajectories,axis=0)

            for key in batch.conditions.keys():
                batch.conditions[key] = torch.tensor(np.expand_dims(batch.conditions[key], axis=0))

            if Args.include_goal_in_state:
                cons = batch.conditions.copy()
                for key in cons.keys():
                    con = cons[key]
                    # copy current condition and repeat if need
                    con = torch.hstack((con, con[:, :Args.repeat_len]))
                    cons[key] = con

                batch = batch._replace(conditions=cons)

            conditions = to_device(batch.conditions, 'cuda:0')

            start_info = np.array(conditions[0].cpu())[:, None]
            if Args.include_goal_in_state:
                start_info = start_info[...,:-Args.repeat_len]
            start_info = self.dataset.normalizer.unnormalize(start_info, 'observations')
            x = start_info[0][0][0]
            y = start_info[0][0][1]
            xv = start_info[0][0][2]
            yv = start_info[0][0][3]
            start_pos = np.array([x, y])
            start_vel = np.array([xv, yv])

            targets_info = np.array(conditions[383].cpu())[:, None]
            if Args.include_goal_in_state:
                targets_info = targets_info[...,:-Args.repeat_len]
            targets_info = self.dataset.normalizer.unnormalize(targets_info,'observations')
            x = targets_info[0][0][0]
            y = targets_info[0][0][1]
            xv = targets_info[0][0][2]
            yv = targets_info[0][0][3]
            target = np.array([x, y])
            all_tar = np.array([x, y, xv, yv])


            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions,horizon=Args.horizon)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            if Args.include_goal_in_state:
                normed_observations = samples[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]
            else:
                normed_observations = samples[:, :, self.dataset.action_dim:]
            normed_actions = samples[:, :, :self.dataset.action_dim]

            # [ 1 x 1 x observation_dim ]
            if Args.include_goal_in_state:
                normed_conditions = to_np(batch.conditions[0])[:, None]
                normed_conditions = normed_conditions[...,:self.dataset.observation_dim]
                normed_reference_observations = trajectories[:, :,
                                      self.dataset.action_dim:self.dataset.action_dim + self.dataset.observation_dim]
                normed_reference_actions = trajectories[:, :,
                                                :self.dataset.action_dim]
            else:
                normed_conditions = to_np(batch.conditions[0])[:,None]
                normed_reference_observations = trajectories[:, :, self.dataset.action_dim:]
                normed_reference_actions = trajectories[:, :,
                                           :self.dataset.action_dim]

            reference_observations = self.dataset.normalizer.unnormalize(normed_reference_observations, 'observations')
            reference_actions=self.dataset.normalizer.unnormalize(normed_reference_actions, 'actions')

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            ## execute actions
            ## [ n_samples x horizon x action_dim ]
            actions_set=self.dataset.normalizer.unnormalize(normed_actions, 'actions')

            '''
            Below to get reference image
            '''

            '''
            # execute the reference observations to see the reward
            ref_img=[]
            # execute the exp with observation changes
            self.env.reset()
            # set initial position
            self.env.set_state(start_pos, start_vel)
            obs = self.env._get_obs()
            # set target position
            self.env.set_target(target)
            total_reward = 0

            exe_obs = [obs.copy()]
            target = self.env._target
            cond = {
                self.model.horizon - 1: np.array([*target, 0, 0]),
            }
            for t in range(reference_observations.shape[1]):
                state = self.env.state_vector().copy()

                if t == 0:
                    cond[0] = obs.copy()
                    sequence = reference_observations[0]
                if t < len(sequence) - 1:
                    next_waypoint = sequence[t + 1]
                else:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0

                ## can use actions or define a simple controller based on state predictions
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                next_obs, reward, terminal, _ = self.env.step(action)
                total_reward += reward
                # score = total_reward
                score = self.env.get_normalized_score(total_reward)
                exe_obs.append(next_obs.copy())
                if t == reference_observations.shape[1] - 1 or terminal:
                    exe_obs.append(all_tar)
                    ref_img.append(exe_obs)
            reference_score = score
            '''

            # execute the reference observations to see the reward
            ref_img = []
            # execute the exp with observation changes
            self.env.reset()
            # set initial position
            self.env.set_state(start_pos, start_vel)
            obs = self.env._get_obs()
            # set target position
            self.env.set_target(target)
            total_reward = 0

            exe_obs = [obs.copy()]
            reference_actions = reference_actions[0]
            for t in range(reference_actions.shape[0]):
                action = reference_actions[t]
                next_obs, reward, terminal, _ = self.env.step(action)
                total_reward += reward
                # score = total_reward
                score = self.env.get_normalized_score(total_reward)
                exe_obs.append(next_obs.copy())
                if t == reference_actions.shape[0] - 1 or terminal:
                    # manully set the final state to be target state
                    exe_obs.append(all_tar)
                    exe_obs.append(all_tar)
                    # save image name processing
                    ref_img.append(exe_obs)
            reference_score = score

            '''
            Below to get predicted image
            '''

            images1 = []
            images2 = []
            for n in range(n_samples):
                # execute the exp with actions
                ## [ n_samples x horizon x action_dim ]-->[ 1 x horizon x action_dim ]
                actions = actions_set[n]
                # reset environment
                self.env.reset()
                # set initial position
                self.env.set_state(start_pos, start_vel)
                obs = self.env._get_obs()
                # set target position
                self.env.set_target(target)
                total_reward = 0

                exe_obs = [obs.copy()]
                for t in range(actions.shape[0]):
                    action = actions[t]
                    next_obs, reward, terminal, _ = self.env.step(action)
                    total_reward += reward
                    # score = total_reward
                    score = self.env.get_normalized_score(total_reward)
                    exe_obs.append(next_obs.copy())
                    if t == actions.shape[0] - 1 or terminal:
                        # manully set the final state to be target state
                        exe_obs.append(all_tar)
                        # save image name processing
                        images1.append(exe_obs)
                exe_act_score.append(score)

                # execute the exp with observation changes
                self.env.reset()
                # set initial position
                self.env.set_state(start_pos, start_vel)
                obs = self.env._get_obs()
                # set target position
                self.env.set_target(target)
                total_reward = 0

                exe_obs = [obs.copy()]
                target = self.env._target
                cond = {
                    self.model.horizon - 1: np.array([*target, 0, 0]),
                }
                for t in range(actions.shape[0]):
                    state = self.env.state_vector().copy()

                    if t == 0:
                        cond[0] = obs.copy()
                        sequence = observations[n]
                    if t < len(sequence) - 1:
                        next_waypoint = sequence[t + 1]
                    else:
                        next_waypoint = sequence[-1].copy()
                        next_waypoint[2:] = 0

                    ## can use actions or define a simple controller based on state predictions
                    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                    next_obs, reward, terminal, _ = self.env.step(action)
                    total_reward += reward
                    score = self.env.get_normalized_score(total_reward)
                    exe_obs.append(next_obs.copy())
                    if t == actions.shape[0] - 1 or terminal:
                        exe_obs.append(all_tar)
                        images2.append(exe_obs)
                exe_obs_score.append(score)

            stp = int(self.step / self.sample_freq)
            index = (str(stp) + 'k').zfill(5)

            log_dict = {
                    # f"reward-{n_samples}": wandb.Histogram(exe_obs_score),
                    "reference-reward": reference_score,
                    "obs.vs.ref": np.array(exe_obs_score).mean() / reference_score,
                    "act.vs.ref": np.array(exe_act_score).mean() / reference_score,
                    "act-reward": np.array(exe_act_score).mean(),
                    "obs-reward": np.array(exe_obs_score).mean(),
                    # "act-reward-var": np.array(exe_act_score).var(),
                    # "obs-reward-var": np.array(exe_obs_score).var(),
                    "case": i,
                }
            wandb.log(log_dict)

            total_obs_mean.append(np.array(exe_obs_score).mean())
            total_obs_var.append(np.array(exe_obs_score).var())
            total_ref_mean.append(reference_score)
            total_act_mean.append(np.array(exe_act_score).mean())
            total_act_var.append(np.array(exe_act_score).var())

            if i % 5 == 0 or np.array(exe_obs_score).mean()/reference_score>1:
                savepath = os.path.join(self.logdir, f'eval/act/{index}-{i}.png')
                con_image1 = self.renderer.composite(savepath, np.array(images1), ncol=5)
                wbimg1=wandb.Image(np.array(con_image1),caption=f'act-{index}-{i}')
                wandb.log({f"act-{i}th": wbimg1})

                savepath = os.path.join(self.logdir, f'eval/obs/{index}-{i}.png')
                con_image2 = self.renderer.composite(savepath, np.array(images2), ncol=5)
                wbimg2 = wandb.Image(np.array(con_image2), caption=f'obs-{index}-{i}')
                wandb.log({f"obs-{i}th": wbimg2})

                savepath = os.path.join(self.logdir, f'eval/pre/{index}-{i}.png')
                images = self.renderer.composite(savepath, observations)
                wbimgpre = wandb.Image(np.array(images), caption=f'pre-{index}-{i}')
                wandb.log({f"pre-{i}th": wbimgpre})

                if not os.path.exists(os.path.join(self.logdir, 'eval/ref')):
                    os.mkdir(os.path.join(self.logdir, 'eval/ref'))

                savepath = os.path.join(self.logdir, f'eval/ref/{index}-{i}.png')
                images = self.renderer.composite(savepath, np.array(ref_img), ncol=1)
                wbimgpre = wandb.Image(np.array(images), caption=f'ref-{index}-{i}')
                wandb.log({f"ref-{i}th": wbimgpre})

        res_dict = {
            "ref-mean": np.array(total_ref_mean).mean(),
            "obs-mean": np.array(total_obs_mean).mean(),
            "obs-var": np.array(total_obs_var).mean(),
            "act-mean": np.array(total_act_mean).mean(),
            "act-var": np.array(total_act_var).mean(),
        }

        wandb.log(res_dict)