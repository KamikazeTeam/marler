from gym.envs.registration import register

register(
    id='PredatorPrey-v0',
    entry_point='games.predator_prey_env:PredatorPreyEnv',
)

register(
    id='TrafficJunction-v0',
    entry_point='games.traffic_junction_env:TrafficJunctionEnv',
)
