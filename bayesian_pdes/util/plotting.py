import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tempfile import NamedTemporaryFile
from .. import parabolic


def monkeypatch_animation():
    video_tag = """<video autoplay loop controls>
     <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
     Your browser does not support the video tag.
    </video>"""

    def anim_to_html(anim):
        if not hasattr(anim, '_encoded_video'):
            with NamedTemporaryFile(suffix='.mp4') as f:
                anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
                video = open(f.name, "rb").read()
            anim._encoded_video = video.encode("base64")

        return video_tag.format(anim._encoded_video)
    animation.Animation._repr_html_ = anim_to_html


def plot_parabolic_1d(x_points, posteriors, times, augment_times=False, stride=1, deterministic=None):
    means, covs = [], []
    times = times[::stride]
    for p, time in zip(posteriors[::stride], times):
        eval_test = x_points if not augment_times else parabolic.augment_with_time(x_points, time)
        mu, cov = p(eval_test)
        means.append(mu)
        covs.append(cov)
    means = np.column_stack(means)

    fig = plt.gcf()
    ax = plt.gca()
    line, = ax.plot([], [], c='blue', label='Mean')
    ub, = ax.plot([], [], linestyle='--', c='black')
    lb, = ax.plot([], [], linestyle='--', c='black')
    if deterministic is not None:
        l_det, = ax.plot([], [], c='green', label='Truth')
        det_points, det_vals = deterministic
        det_vals = det_vals[::stride]
    plt.legend()

    def init():
        return []

    def animate(i):
        cov = covs[i]
        mean = means[:,i]
        line.set_data(x_points, mean)
        ub.set_data(x_points, mean+np.diag(cov))
        lb.set_data(x_points, mean-np.diag(cov))
        ax.set_title('t={}'.format(times[i]))
        if deterministic is not None:
            l_det.set_data(det_points, det_vals[i])
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=means.shape[1], interval=20, blit=False)
    plt.close()
    return anim