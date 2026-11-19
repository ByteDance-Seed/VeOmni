import enum
import math
from typing import Callable

import numpy as np
import torch as th

from . import path
from .integrators import ode, sde
from .utils import expand_dims, mean_flat


class ModelType(enum.Enum):
    """Which type of output the model predicts."""

    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    """Which type of path to use."""

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """Which type of weighting to use."""

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps, snr_type, do_shift, seq_len):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        self.snr_type = snr_type
        self.do_shift = do_shift
        self.seq_len = seq_len

    def prior_logp(self, z):
        """Standard multivariate normal prior assuming z is batched."""
        shape = th.tensor(z.size())
        n_dim = th.prod(shape[1:])

        def _fn(x):
            return -n_dim / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0

        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sample x0 and t based on shape of x1."""
        if isinstance(x1, (list, tuple)):
            x0 = [th.randn_like(img_start) for img_start in x1]
        else:
            x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if self.snr_type.startswith("uniform"):
            assert t0 == 0.0 and t1 == 1.0, "not implemented."
            if "_" in self.snr_type:
                _, t0, t1 = self.snr_type.split("_")
                t0, t1 = float(t0), float(t1)
            t = th.rand((len(x1),)) * (t1 - t0) + t0
        elif self.snr_type == "lognorm":
            u = th.normal(mean=0.0, std=1.0, size=(len(x1),))
            t = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
        else:
            raise NotImplementedError(f"Not implemented snr_type {self.snr_type}")

        if self.do_shift:
            base_shift: float = 0.5
            max_shift: float = 1.15
            mu = self.get_lin_function(y1=base_shift, y2=max_shift)(self.seq_len)
            t = self.time_shift(mu, 1.0, t)
        t = t.to(x1[0])
        return t, x0, x1

    def time_shift(self, mu: float, sigma: float, t: th.Tensor):
        t = 1 - t
        t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
        t = 1 - t
        return t

    def get_lin_function(
        self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def training_losses(self, model, x1, model_kwargs=None):
        """Loss for training the score model."""
        if model_kwargs is None:
            model_kwargs = {}
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        if "cond" in model_kwargs:
            conds = model_kwargs.pop("cond")
            xt = [th.cat([x, cond], dim=0) if cond is not None else x for x, cond in zip(xt, conds)]
        model_output = model(xt, t, **model_kwargs)

        if hasattr(model_output, "sample"):
            model_output = model_output.sample
        elif isinstance(model_output, tuple):
            model_output = model_output[0]

        batch_size = len(x0)

        terms = {}
        if self.model_type == ModelType.VELOCITY:
            if isinstance(x1, (list, tuple)):
                assert len(model_output) == len(ut) == len(x1)
                for i in range(batch_size):
                    assert model_output[i].shape == ut[i].shape == x1[i].shape, (
                        f"{model_output[i].shape} {ut[i].shape} {x1[i].shape}"
                    )
                terms["task_loss"] = th.stack(
                    [((ut[i] - model_output[i]) ** 2).mean() for i in range(batch_size)],
                    dim=0,
                )
            else:
                terms["task_loss"] = mean_flat((model_output - ut) ** 2)
        else:
            raise NotImplementedError

        terms["loss"] = terms["task_loss"]
        terms["task_loss"] = terms["task_loss"].clone().detach()
        terms["t"] = t
        return terms

    def get_drift(self):
        """Obtain the drift of the probability flow ODE."""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(self):
        """Obtain score of x_t = alpha_t * x + sigma_t * eps."""
        if self.model_type == ModelType.NOISE:

            def score_fn(x, t, model, **kwargs):
                return model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]

        elif self.model_type == ModelType.SCORE:

            def score_fn(x, t, model, **kwargs):
                return model(x, t, **kwargs)

        elif self.model_type == ModelType.VELOCITY:

            def score_fn(x, t, model, **kwargs):
                return self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)

        else:
            raise NotImplementedError

        return score_fn


class Sampler:
    """Sampler class for the transport model."""

    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)

        def sde_drift(x, t, model, **kwargs):
            return self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        return sde_drift, diffusion_fn

    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        """Get the last step function of the SDE solver."""
        if last_step is None:

            def last_step_fn(x, t, model, **model_kwargs):
                del t, model, model_kwargs
                return x

        elif last_step == "Mean":

            def last_step_fn(x, t, model, **model_kwargs):
                return x + sde_drift(x, t, model, **model_kwargs) * last_step_size

        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t

            def last_step_fn(x, t, model, **model_kwargs):
                return x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(
                    x, t, model, **model_kwargs
                )

        elif last_step == "Euler":

            def last_step_fn(x, t, model, **model_kwargs):
                return x + self.drift(x, t, model, **model_kwargs) * last_step_size

        else:
            raise NotImplementedError

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            assert len(xs) == num_steps, "Samples does not match the number of steps"
            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        do_shift=False,
        time_shifting_factor=None,
        stochast_ratio=0.0,
    ):
        if stochast_ratio == 0.0:

            def drift(x, t, model, **kwargs):
                return self.drift(x, t, model, **kwargs)

            t0, t1 = self.transport.check_interval(
                self.transport.train_eps,
                self.transport.sample_eps,
                sde=False,
                eval=True,
                reverse=reverse,
                last_step_size=0.0,
            )
            _ode = ode(
                drift=drift,
                t0=t0,
                t1=t1,
                sampler_type=sampling_method,
                num_steps=num_steps,
                atol=atol,
                rtol=rtol,
                do_shift=do_shift,
                time_shifting_factor=time_shifting_factor,
            )
            return _ode.sample

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )
        path_sampler = self.transport.path_sampler

        def _sample(init, model, **model_kwargs):
            t_steps = th.linspace(t0, t1, num_steps + 1, dtype=th.float64).to(init)
            x_cur = init.to(th.float64)

            for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
                t_batch = th.ones(x_cur.size(0), device=x_cur.device, dtype=x_cur.dtype) * t_cur
                v = model(x_cur, t_batch, **model_kwargs)

                t_exp = expand_dims(t_batch, x_cur.dim())
                alpha_t, d_alpha_t = path_sampler.compute_alpha_t(t_exp)
                sigma_t, d_sigma_t = path_sampler.compute_sigma_t(t_exp)
                denom = sigma_t * d_alpha_t - d_sigma_t * alpha_t
                x1_hat = (sigma_t * v - d_sigma_t * x_cur) / denom
                x0_hat = (d_alpha_t * x_cur - alpha_t * v) / denom

                t_next_batch = th.ones_like(t_batch) * t_next
                t_next_exp = expand_dims(t_next_batch, x_cur.dim())
                alpha_next, _ = path_sampler.compute_alpha_t(t_next_exp)
                sigma_next, _ = path_sampler.compute_sigma_t(t_next_exp)

                noi = th.randn_like(x_cur)
                x_cur = alpha_next * x1_hat + sigma_next * (
                    x0_hat * ((1 - stochast_ratio) ** 0.5) + noi * (stochast_ratio**0.5)
                )

            return [x_cur]

        return _sample
