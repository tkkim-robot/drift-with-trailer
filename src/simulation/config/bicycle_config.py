from dataclasses import dataclass

@dataclass(slots=True)
class TrackConfig:
    file = "ADD PATH" # TODO
    width = 16.0
    # progress bins seems uncecessary, check


@dataclass(slots=True)
class VehicleConfig:
    wheelbase = 3.05
    lf = 1.45
    lr = 1.6
    mass = 720.0
    inertia_z = 900.0
    cornering_stiffness_front = 90000.0
    cornering_stiffness_rear = 98000.0
    max_steer_rad = 0.32
    max_accel = 12.0
    max_brake = 18.0
    drag_coefficient = 0.85
    wheel_radius = 0.33
    chassis_size = [3.2, 1.4, 0.32]

@dataclass(slots=True)
class SimulationConfig:
    dt = 0.05
    lookahead_points: 6
    lookahead_spacing_m: 10.0

@dataclass(slots=True)
class BicycleEnvConfig:
    name: str
    track: TrackConfig
    vehicle: VehicleConfig
    simulation: SimulationConfig
