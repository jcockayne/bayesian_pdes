from autograd import numpy as np
from util.linalg import schur
import collocation
import logging

logger = logging.getLogger(__name__)


def augment_with_time(spatial_points, time):
    return np.column_stack([spatial_points, time * np.ones((spatial_points.shape[0], 1))])


def solve_parabolic(op_system, ops, ops_bar, times, obs_function, ics, fun_args=None):

    all_obs = [ics]
    all_ops = [[()]]
    all_ops_bar = [[()]]

    # TODO: this is all assuming that we are observing directly the solution at t_0
    init_ops = [()]
    init_ops_bar = [()]

    K_0 = collocation.calc_LLbar(init_ops, init_ops_bar, ics, op_system, fun_args)
    last_inv = np.linalg.inv(K_0)

    posterior_t = collocation.CollocationPosterior(flatten_list(all_ops),
                                                   flatten_list(all_ops_bar),
                                                   op_system,
                                                   flatten_list(all_obs),
                                                   last_inv,
                                                   fun_args=fun_args)

    posteriors = [posterior_t]
    for t in times[1:]:

        logger.info('t={}'.format(t))
        obs_t = obs_function(t)

        all_obs.append(obs_t)
        all_ops.append(ops)
        all_ops_bar.append(ops_bar)

        posterior_t = step_forward(all_ops, all_ops_bar, all_obs, op_system, fun_args, posteriors[-1])
        posteriors.append(posterior_t)

    return posteriors

def step_forward(all_ops, all_ops_bar, all_obs, op_system, fun_args, last_posterior):
    ops, ops_bar, obs_t = all_ops[-1], all_ops_bar[-1], all_obs[-1]
    prev_inv = last_posterior.__LLbar_inv__

    K_t = collocation.calc_LLbar(ops, ops_bar, obs_t, op_system, fun_args)
    K_t_inv = np.linalg.inv(K_t)

    K_ttm1 = []
    for obs, op in zip(all_obs[:-1], all_ops[:-1]):
        K_ttm1.append(calc_diff_LLbar(op, ops_bar, obs, obs_t, op_system, fun_args=fun_args))

    logger.debug('Concatenating K_ttm1; shapes are [{}]'.format([k.shape for k in K_ttm1]))
    K_ttm1 = np.concatenate(K_ttm1)
    logger.debug('About to call schur; shapes are: last_inv: {} K_ttm1: {}, K_t: {}; K_t_inv: {}'.format(
        prev_inv.shape,
        K_ttm1.shape,
        K_t.shape,
        K_t_inv.shape
    ))
    new_inv = schur(prev_inv, K_ttm1, K_t, K_t_inv)

    posterior_t = collocation.CollocationPosterior(flatten_list(all_ops),
                                                   flatten_list(all_ops_bar),
                                                   op_system,
                                                   flatten_list(all_obs),
                                                   new_inv,
                                                   fun_args=fun_args)

    return posterior_t


def flatten_list(list):
    return [item for l in list for item in l]


def calc_diff_LLbar(ops, ops_bar, obs_1, obs_2, op_system, fun_args):
    logger.debug('ops has len {}; obs_1 has len {}; ops_bar has len {}; obs_2 has len {}'.format(
        len(ops),
        len(obs_1),
        len(ops_bar),
        len(obs_2)
    ))
    # and build the 2D matrix
    LLbar = []

    for op, o1 in zip(ops, obs_1):
        tmp = []
        for op_bar, o2 in zip(ops_bar, obs_2):
            points_1, _ = o1
            points_2, _ = o2

            fun_op = op_system[(op, op_bar)]

            applied = fun_op(points_1, points_2, fun_args)
            tmp.append(applied)
        LLbar.append(np.concatenate(tmp, axis=1))

    return np.concatenate(LLbar)
