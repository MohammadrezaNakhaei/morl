"""
Launcher for experiments with CSRO

"""
import os

os.environ['MUJOCO_PY_MJPRO_PATH'] = '~/.mujoco/mujoco210'

import pathlib
import numpy as np
import click
import json
import torch
import random
import multiprocessing as mp
from itertools import product

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, MlpQuantizedEncoder
from rlkit.torch.sac.sac import CSROSoftActorCritic, CSROPrediction
from rlkit.torch.sac.mutual_info_reduction import MIRSoftActorCritic
from rlkit.torch.sac.mutual_estimator_reduction import MIERSoftActorCritic
from rlkit.torch.sac.mi_estimator import MIKernelEstimator
from rlkit.torch.sac.csro_gan import CSROGAN, CSROCondGAN
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
import pdb



def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def experiment(variant, seed=None):

    algorithm = initialize(variant, seed)

    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else variant['util_params']['exp_name']
    experiment_log_dir, wandb_logger = setup_logger(
        variant['env_name'],
        variant=variant,
        exp_id=exp_id,
        base_log_dir=variant['util_params']['base_log_dir'],
        seed=seed,
        snapshot_mode="gap_and_last",
        snapshot_gap=5
    )

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train(wandb_logger)
    wandb_logger.finish()

def initialize(variant, seed=None):
    # create multi-task environment and sample tasks, normalize obs if provided with 'normalizer.npz'
    if 'normalizer.npz' in os.listdir(variant['algo_params']['data_dir']):
        obs_absmax = np.load(os.path.join(variant['algo_params']['data_dir'], 'normalizer.npz'))['abs_max']
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']), obs_absmax=obs_absmax)
    else:
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))

    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        output_activation=torch.tanh,
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    # critic network for divergence in dual form (see BRAC paper https://arxiv.org/abs/1911.11361)
    c = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1
    )

    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    task_modes = env.task_modes()

    if variant['algo_type'] == 'CSRO':
        if variant['algo_params']['club_use_sa']:
            club_input_dim = obs_dim + action_dim
        else:
            club_input_dim = obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim

        club_model = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=club_input_dim,
            output_size=latent_dim * 2,
            output_activation=torch.tanh,
            # output_activation_half=True
        )

        algorithm = CSROSoftActorCritic(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, club_model],
            latent_dim=latent_dim,
            **variant['algo_params']
        )

    if variant['algo_type'] == 'CSROPrediction':
        if variant['algo_params']['club_use_sa']:
            club_input_dim = obs_dim + action_dim
        else:
            club_input_dim = obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim

        club_model = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=club_input_dim,
            output_size=latent_dim * 2,
            output_activation=torch.tanh,
            # output_activation_half=True
        )

        decoder = FlattenMlp(
            hidden_sizes = [net_size, net_size, net_size],
            input_size = obs_dim + action_dim + latent_dim,
            output_size= obs_dim + 1,
        )

        algorithm = CSROPrediction(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, club_model],
            latent_dim=latent_dim,
            decoder=decoder,
            **variant['algo_params']
        )

    if variant['algo_type'] == 'MIR':
        behavior_policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        decoder = FlattenMlp(
            hidden_sizes = [net_size, net_size, net_size],
            input_size = obs_dim + action_dim + latent_dim,
            output_size= obs_dim + 1,
        )
        algorithm = MIRSoftActorCritic(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, behavior_policy, decoder],
            latent_dim=latent_dim,
            **variant['algo_params']
        )

    if variant['algo_type'] == 'MIER': # mutual information estimator reduction
        mi_estimator = MIKernelEstimator(
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
            number_of_samples = variant['algo_params']['embedding_batch_size']//8,
            x_size = obs_dim+action_dim,
            y_size = latent_dim,
            use_joint = True,
        )
        decoder = FlattenMlp(
            hidden_sizes = [net_size, net_size, net_size],
            input_size = obs_dim + action_dim + latent_dim,
            output_size= obs_dim + 1,
        )
        algorithm = MIERSoftActorCritic(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, mi_estimator, decoder],
            latent_dim=latent_dim,
            **variant['algo_params']
        )

    if variant['algo_type'] == 'GAN':
        generator = FlattenMlp(
            hidden_sizes = [200, 200, 200],
            input_size = latent_dim+obs_dim+variant['algo_params']['generator_dim'],
            output_size = action_dim,
            output_activation = torch.tanh
         )
        discriminator = FlattenMlp(
            hidden_sizes = [net_size,],
            input_size = action_dim,
            output_size = 1,
            output_activation = torch.sigmoid,
        )

        algorithm = CSROGAN(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, generator, discriminator],
            latent_dim=latent_dim,
            **variant['algo_params']
        )

    if variant['algo_type'] == 'CondGAN':
        generator = FlattenMlp(
            hidden_sizes = [200, 200, 200],
            input_size = latent_dim+obs_dim+variant['algo_params']['generator_dim'],
            output_size = action_dim,
            output_activation = torch.tanh
         )
        discriminator = FlattenMlp(
            hidden_sizes = [net_size, net_size],
            input_size = action_dim+obs_dim+latent_dim,
            output_size = 1,
            output_activation = torch.sigmoid,
        )

        algorithm = CSROCondGAN(
            env=env,
            train_tasks=task_modes['train'],
            eval_tasks=task_modes['moderate'],
            extreme_tasks=task_modes['extreme'],
            nets=[agent, qf1, qf2, vf, c, generator, discriminator],
            latent_dim=latent_dim,
            **variant['algo_params']
        )
    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        epoch = variant['epoch_to_weights']
        agent_ckpt = torch.load(os.path.join(path, "seed"+str(seed), f'agent_itr_{epoch}.pth'))
        club_model.load_state_dict(agent_ckpt['club_model'])
        context_encoder.load_state_dict(agent_ckpt['context_encoder'])
        qf1.load_state_dict(agent_ckpt['qf1'])
        qf2.load_state_dict(agent_ckpt['qf2'])
        vf.load_state_dict(agent_ckpt['vf'])
        algorithm.networks[-3].load_state_dict(agent_ckpt['target_vf'])
        policy.load_state_dict(agent_ckpt['policy'])
        c.load_state_dict(agent_ckpt['c'])

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    return algorithm


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--seed', default=0)
@click.option('--exp_name', default=None)
def main(config, gpu, seed=0, exp_name=None):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    if not (exp_name == None):
        variant['util_params']['exp_name'] = exp_name

    # multi-processing
    experiment(variant, seed)

if __name__ == "__main__":
    main()
