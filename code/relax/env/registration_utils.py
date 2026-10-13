from gymnasium.envs.registration import register, registry


def safe_register(id: str, **kwargs):
    if id in registry:
        return
    register(id=id, **kwargs)
