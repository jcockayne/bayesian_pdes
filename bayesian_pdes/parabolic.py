from autograd import numpy as np
from util.linalg import schur
import collocation
import logging
logger = logging.getLogger(__name__)


def solve_parabolic(op_system, times, obs_function, test_function, ics, fun_args=None):
    ops = op_system.operators
    ops_bar = op_system.operators_bar

    test_t = test_function(times[0])
    all_obs = [ics]

    # TODO: this is all assuming that we are observing directly the solution at t_0
    init_ops = [()]
    init_ops_bar = [()]

    K_0 = collocation.calc_LLbar(init_ops, init_ops_bar, ics, op_system, fun_args)
    last_inv = np.linalg.inv(K_0)

    right, left = collocation.calc_side_matrices(
        ops,
        ops_bar,
        ics,
        test_t,
        op_system,
        fun_args)
    kern = op_system[()](test_t, test_t, fun_args)
    covs = [kern - np.dot(left, np.dot(last_inv, right))]
    rhs = ics[0][1]
    means = [np.dot(left, np.dot(last_inv, rhs))]

    for t in times[1:]:
        logger.info('t={}'.format(t))
        obs_t = obs_function(t)

        test_t = test_function(t)

        K_t = collocation.calc_LLbar(ops, ops_bar, obs_t, op_system, fun_args)
        K_t_inv = np.linalg.inv(K_t)

        K_ttm1 = []
        rmat = []
        lmat = []
        for ix, o in enumerate(all_obs):
            this_ops, this_ops_bar = (ops, ops_bar) if ix != 0 else (init_ops, init_ops_bar)
            right, left = collocation.calc_side_matrices(
                this_ops,
                this_ops_bar,
                o,
                test_t,
                op_system,
                fun_args)
            rmat.append(right)
            lmat.append(left)
            K_ttm1.append(calc_diff_LLbar(this_ops, ops_bar, o, obs_t, op_system, fun_args))

        logger.debug('Concatenating K_ttm1; shapes are [{}]'.format([k.shape for k in K_ttm1]))
        K_ttm1 = np.concatenate(K_ttm1)
        logger.debug('About to call schur; shapes are: last_inv: {} K_ttm1: {}, K_t: {}; K_t_inv: {}'.format(
            last_inv.shape,
            K_ttm1.shape,
            K_t.shape,
            K_t_inv.shape
        ))
        new_inv = schur(last_inv, K_ttm1, K_t, K_t_inv)

        right, left = collocation.calc_side_matrices(
            ops,
            ops_bar,
            obs_t,
            test_t,
            op_system,
            fun_args)
        rmat.append(right)
        lmat.append(left)
        right = np.concatenate(rmat)
        left = np.concatenate(lmat, axis=1)

        all_obs.append(obs_t)

        kern = op_system[()](test_t, test_t, fun_args)
        rhs = np.concatenate([np.concatenate([v for p,v in o]) for o in all_obs])
        means.append(np.dot(left, np.dot(new_inv, rhs)))

        new_cov = kern - np.dot(left, np.dot(new_inv, right))
        covs.append(new_cov)

        last_inv = new_inv

    return means, covs


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