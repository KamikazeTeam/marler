from gym.envs.registration import register

register(
    id='PredatorPrey-v0',
    entry_point='games.predator_prey_env:PredatorPreyEnv',
    disable_env_checker=True,  #
)

register(
    id='TrafficJunction-v0',
    entry_point='games.traffic_junction_env:TrafficJunctionEnv',
)

register(
    id='TalentLuck-v0',
    entry_point='games.talentluck:TalentLuck',
    disable_env_checker=True,  #
)
