from .. import collocation
import logging

logger = logging.getLogger(__name__)


def solve_theta(time_ops,
                other_ops,
                op_system,
                time_design_function,
                other_obs_function,
                ics,
                times,
                theta,
                fun_args,
                return_key='all',
                debug=False):
    # assumption: times equally spaced
    dt = times[1] - times[0]
    parabolic_op_system = ParabolicOperatorSystem(op_system, time_ops[0], time_ops[1], theta, dt)

    # todo: assumes initial observation is directly of the solution
    initial_posterior = collocation.collocate([()], [()], ics, parabolic_op_system, fun_args)
    cur_posterior = initial_posterior

    if return_key == 'all':
        posteriors = [cur_posterior]

    # todo
    push_forward_ops, push_forward_ops_bar = [parabolic_op_system.explicit_op], [parabolic_op_system.explicit_op_bar]
    implicit_ops = [parabolic_op_system.implicit_op] + other_ops[0]
    implicit_ops_bar = [parabolic_op_system.implicit_op_bar] + other_ops[1]

    for t, tm1 in zip(times[1:], times[:-1]):
        # TODO: assumes design constant over all time?
        time_points = time_design_function(t)
        last_time_points = time_design_function(tm1)

        # first compute the observations
        operated_int = cur_posterior.apply_operators(push_forward_ops, push_forward_ops_bar)
        mu, Sigma = operated_int(last_time_points)

        #mu_err, Sigma_err = error_dist(x_pts, cur_posterior, t, dt, fun_args)
        mu_err, Sigma_err = 0., 0.

        time_obs = [(time_points, mu + mu_err, Sigma + Sigma_err)]
        extra_obs = other_obs_function(t)

        all_obs = time_obs + extra_obs

        cur_posterior = collocation.collocate(implicit_ops, implicit_ops_bar, all_obs, parabolic_op_system, fun_args)
        if return_key == 'all':
            posteriors.append(cur_posterior)

    return posteriors if return_key == 'all' else cur_posterior


class ParabolicOperatorSystem(object):
    def __init__(self, base_op_system, time_op, time_op_bar, theta, dt):
        self.explicit_op = 'explicit'
        self.explicit_op_bar = 'explicit_bar'
        self.implicit_op = 'implicit'
        self.implicit_op_bar = 'implicit_bar'
        self.time_op = time_op
        self.time_op_bar = time_op_bar
        self.theta = theta
        self.dt = dt
        self.base_op_system = base_op_system

    @property
    def operators(self):
        return [self.implicit_op, self.explicit_op] + self.base_op_system.operators

    @property
    def operators_bar(self):
        return [self.implicit_op_bar, self.explicit_op_bar] + self.base_op_system.operators_bar

    def __getitem__(self, item):
        if type(item) is not tuple:
            item = (item,)
        item = remove_identity(item)
        non_special_cases = remove_operators(item, [self.explicit_op, self.implicit_op, self.explicit_op_bar, self.implicit_op_bar])

        if non_special_cases == item:
            return self.base_op_system[item]

        non_special = remove_operators(non_special_cases, self.operators_bar)
        barred_non_special = remove_operators(non_special_cases, self.operators)

        time_op = self.base_op_system[(self.time_op, ) + barred_non_special]
        time_op_bar = self.base_op_system[(self.time_op_bar, ) + non_special]
        double_time_op = self.base_op_system[(self.time_op, self.time_op_bar)]
        ident_op = self.base_op_system[non_special_cases]

        # handle double ops
        if item == (self.explicit_op, self.explicit_op_bar):
            return get_double_explicit_function(ident_op, time_op, time_op_bar, double_time_op, self.theta, self.dt)
        if item == (self.implicit_op, self.implicit_op_bar):
            return get_double_implicit_function(ident_op, time_op, time_op_bar, double_time_op, self.theta, self.dt)
        if item == (self.explicit_op, self.implicit_op_bar):
            return get_explicit_implicit_function(ident_op, time_op, time_op_bar, double_time_op, self.theta, self.dt)
        if item == (self.implicit_op, self.explicit_op_bar):
            return get_implicit_explicit_function(ident_op, time_op, time_op_bar, double_time_op, self.theta, self.dt)

        special = remove_operators(item, non_special_cases)
        # handle single ops
        if special == (self.explicit_op,):
            return get_explicit_function(ident_op, time_op, self.theta, self.dt)
        if special == (self.explicit_op_bar,):
            return get_explicit_function(ident_op, time_op_bar, self.theta, self.dt)
        if special == (self.implicit_op,):
            return get_implicit_function(ident_op, time_op, self.theta, self.dt)
        if special == (self.implicit_op_bar,):
            return get_implicit_function(ident_op, time_op_bar, self.theta, self.dt)

        raise Exception('Could not interpret operator {}'.format(item))

def remove_identity(item):
    return remove_operators(item, [()])


def remove_operators(item, operators):
    non_operator = [k for k in item if k not in operators]
    return tuple(non_operator)


def get_explicit_function(ident_op, time_op, theta, dt):
    def __ret__(x, y, args):
        return ident_op(x, y, args) + theta*dt*time_op(x, y, args)
    return __ret__


def get_double_explicit_function(ident_op, time_op, time_op_bar, double_time_op, theta, dt):
    def __ret__(x, y, args):
        return ident_op(x, y, args) \
               + theta*dt*time_op(x, y, args) \
               + theta*dt*time_op_bar(x, y, args) \
               + theta**2*dt**2*double_time_op(x, y, args)
    return __ret__


def get_implicit_function(ident_op, time_op, theta, dt):
    def __ret__(x, y, args):
        return ident_op(x, y, args) - (1-theta)*dt*time_op(x, y, args)
    return __ret__


def get_double_implicit_function(ident_op, time_op, time_op_bar, double_time_op, theta, dt):
    def __ret__(x, y, args):
        return ident_op(x, y, args) \
               - (1-theta)*dt*time_op(x, y, args) \
               - (1-theta)*dt*time_op_bar(x, y, args) \
               + (1-theta)**2*dt**2*double_time_op(x, y, args)
    return __ret__


def get_explicit_implicit_function(ident_op, time_op, time_op_bar, double_time_op, theta, dt):
    def __ret__(x, y, args):
        return ident_op(x, y, args) \
               + theta*dt*time_op(x, y, args) \
               - (1-theta)*dt*time_op_bar(x, y, args) \
               - theta*(1-theta)*dt**2*double_time_op(x, y, args)
    return __ret__


def get_implicit_explicit_function(ident_op, time_op, time_op_bar, double_time_op, theta, dt):
    def __ret__(x, y, args):
        return ident_op(x, y, args) \
               - (1-theta)*dt*time_op(x, y, args) \
               + theta*dt*time_op_bar(x, y, args) \
               - theta*(1-theta)*dt**2*double_time_op(x, y, args)
    return __ret__
