from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax
from typing import NamedTuple
from src.simulation.config.trailer_bicycle_config import TrailerBicycleEnvConfig
from src.utils.track import TrackModel

Array = jax.Array


class TrackProjection(NamedTuple):
    progress: Array
    arc_length: Array
    x: Array
    y: Array
    heading: Array
    lateral_error: Array
    curvature: Array


def gen_util_funs(
    params: TrailerBicycleEnvConfig,
    reverse=False,
    v_target=None,
    p_weight=1e4,
    p_slow_weight=1e0,
    c_weight=1e-2,
    a_weight=1e5,
):
    reverse = 1 if reverse else -1
    step = params.simulation.dt

    track = TrackModel.from_config(params.track)

    def _project_to_track(x, y, guess) -> tuple[TrackProjection, jax.Array]:
        """
        From Uncertain Racecar Gym, adapted
        """
        WINDOW = 10
        if guess is not None:
            window = ((guess - WINDOW / 2).astype(jnp.int32)) + jnp.arange(WINDOW)
        else:
            window = jnp.arange(len(track.centerline))

        segments_window = jnp.take(track._segments, window, mode="wrap", axis=0)
        segments_len_window = jnp.take(track._segment_lengths, window, mode="wrap")
        segments_sq_window = jnp.take(track._segment_length_sq, window, mode="wrap")
        centerline_window = jnp.take(track.centerline, window, mode="wrap", axis=0)
        segments_normal_window = jnp.take(track._segment_normals, window, mode="wrap", axis=0)
        segments_heading_window = jnp.take(track._segment_headings, window, mode="wrap")
        valid_window = jnp.take(track._segment_valid, window, mode="wrap")
        cumulative_window = jnp.take(track._cumulative, window, mode="wrap")

        point = jnp.stack([x, y])
        delta_from_start = point - centerline_window

        denom = jnp.where(valid_window, segments_sq_window, 1.0)
        t = jnp.where(
            valid_window,
            jnp.einsum("ij,ij->i", delta_from_start, segments_window) / denom,
            0.0,
        )
        t = jnp.clip(t, 0.0, 1.0)
        projected = centerline_window + segments_window * t[:, None]
        delta = point - projected
        distance_sq = jnp.einsum("ij,ij->i", delta, delta)
        distance_sq = jnp.where(valid_window, distance_sq, jnp.inf)
        index = jnp.argmin(distance_sq)  # stays traced; dynamic indexing -> gather

        signed_offset = jnp.dot(point - projected[index], segments_normal_window[index])
        arc = cumulative_window[index] + t[index] * segments_len_window[index]
        return (
            TrackProjection(
                progress=track.arc_to_progress(arc),
                arc_length=arc,
                x=projected[index, 0],
                y=projected[index, 1],
                heading=segments_heading_window[index],
                lateral_error=signed_offset,
                curvature=jnp.interp(
                    arc, track._arc_samples, track._curvature_samples, period=track.length
                ),
            ),
            window[index],
        )

    @jax.jit
    def dynamics(state, u):
        x, y, phi_1, phi_2, v, mu, arc_len = state

        vehicle = params.vehicle
        steer_cmd = jnp.clip(u[0], -1.0, 1.0)
        accel_cmd = jnp.clip(u[1], -1.0, 1.0)

        throttle = jnp.maximum(accel_cmd, 0.0)
        brake = -jnp.minimum(accel_cmd, 0.0)

        commanded = throttle * vehicle.max_accel - brake * vehicle.max_brake

        dt = params.simulation.dt

        throttle = jnp.maximum(accel_cmd, 0.0)
        brake = -jnp.minimum(accel_cmd, 0.0)

        delta = steer_cmd * vehicle.max_steer_rad

        x_dot = v * jnp.cos(phi_1)
        y_dot = v * jnp.sin(phi_1)

        theta1_dot = (v / (vehicle.lf + vehicle.lr)) * jnp.tan(delta)
        theta2_dot = (v / (vehicle.l2f + vehicle.l2r)) * (
            jnp.sin(phi_1 - phi_2)
            - (vehicle.hitch_offset / (vehicle.lf + vehicle.lr))
            * jnp.cos(phi_1 - phi_2)
            * jnp.tan(delta)
        )

        index = jnp.searchsorted(track._cumulative, arc_len, side="right") - 1

        projection_curr, _ = _project_to_track(x, y, index)
        projection_next, _ = _project_to_track(x + dt * x_dot, y + dt * y_dot, index)

        raw_diff = projection_next.arc_length - projection_curr.arc_length
        track_vel = (raw_diff - track.length * jnp.round(raw_diff / track.length)) / dt

        return jnp.array([x_dot, y_dot, theta1_dot, theta2_dot, commanded, 0, track_vel])

    @jax.jit
    def cost(x, u, t):
        
        # Tunable values
        yaw = x[2]
        gvx = x[4] * jnp.cos(yaw)
        gvy = x[4] * jnp.sin(yaw)

        index = jnp.searchsorted(track._cumulative, x[6], side="right") - 1

        projection_curr, _ = _project_to_track(x[0], x[1], index)
        projection_next, _ = _project_to_track(x[0] + step * gvx, x[1] + step * gvy, index)

        raw_diff = projection_next.arc_length - projection_curr.arc_length
        track_vel = (raw_diff - track.length * jnp.round(raw_diff / track.length)) / step

        violation = jnp.maximum(
            0, jnp.abs(projection_curr.lateral_error) - (params.track.width * 0.5) * 0.9 + 0.1
        )

        def wrap_angle(angle):
            return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

        hitch_angle = wrap_angle(x[2] - x[3])

        violation += jnp.maximum(0, jnp.abs(hitch_angle) - params.vehicle.max_hitch)

        if v_target is None:
            v_term = reverse * p_weight * jnp.abs(track_vel) * jnp.sign(x[4])
        else:
            v_term = p_weight * jnp.abs(v_target + reverse * jnp.abs(track_vel) * jnp.sign(x[4]))
            # v_baseline = jnp.minimum(max_safe_v, v_target)
            # # If v is above threshold use actual car velocity instead of track velocity to stop cheating
            # v_car = jnp.where(nominal_v > max_safe_v, nominal_v, track_vel)

            # v_term = p_weight * jnp.maximum(
            #     0, v_car - v_baseline
            # ) + p_weight * p_slow_weight * jnp.maximum(0, v_baseline - v_car)

        c = (
            0.99**t * (1e12 * violation)
            + v_term
            + projection_curr.lateral_error**2 * c_weight
            + jnp.abs(hitch_angle) * a_weight
        )

        # jax.debug.print("cost {c}", c=c)
        return c

    def bound(u):
        return jnp.clip(u, jnp.array([-1, -1]), jnp.array([1, 1]))

    def bound_der(u):
        return u

    return dynamics, cost, bound, bound_der
