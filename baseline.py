import os

import numpy as np
import pandas as pd

from flow.controllers.base_controller import BaseController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.envs.ring.accel import AccelEnv
from flow.networks.ring import RingNetwork


class NaiveBaselineController(BaseController):
    def __init__(self, veh_id, v0=30, s0=20, car_following_params=None):
        """
        veh_id: unique vehicle identifier
        v0: desirable velocity, in m/s
        s0: headway threshold before braking, in m
        """

        BaseController.__init__(self, veh_id, car_following_params)
        self.v0 = v0
        self.s0 = s0

    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        if h < 10:  # car in front less than 10m away
            return -2
        elif v < self.v0:
            return 2
        else:
            return 0

exp_name = "ring_network_baseline"

num_vehicles = 20

vehicles = VehicleParams()
vehicles.add(
    "robot",
    acceleration_controller=(NaiveBaselineController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=num_vehicles,
)

net_params = NetParams(additional_params={
    'length': 200*np.pi,  # 200m diameter
    'lanes': 1,
    'speed_limit': 30,
    'resolution': 40,
})

# initial_config = InitialConfig(spacing="uniform", perturbation=3)
initial_config = InitialConfig(spacing="random", min_gap=2)

env_params = EnvParams(
    additional_params={
        'max_accel': 2,
        'max_decel': 2,
        'target_velocity': 30,
        'sort_vehicles': False,
    },
    horizon=1000,  # number of time steps
)

sim_params = SumoParams(sim_step=0.1, render=True, emission_path='data')

flow_params = {
    'exp_tag': exp_name,
    'env_name': AccelEnv,
    'network': RingNetwork,
    'simulator': 'traci',
    'sim': sim_params,
    'env': env_params,
    'net': net_params,
    'veh': vehicles,
    'initial': initial_config,
}

exp = Experiment(flow_params)

# run the sumo simulation
_ = exp.run(1, convert_to_csv=True)

emission_location = os.path.join(exp.env.sim_params.emission_path, exp.env.network.name)
print(emission_location)

data = pd.read_csv(emission_location + '-0_emission.csv')
print(data.head())
