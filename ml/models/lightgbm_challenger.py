"""Small pooled LightGBM challenger for the historical forecasting MVP.

This factory creates one shared regressor.  It does not fit, evaluate, split,
or transform feature data; those responsibilities remain with the frozen
Day-04 feature artifact and the Day-05 chronological evaluation workflow.
"""

from lightgbm import LGBMRegressor


def build_lightgbm_challenger() -> LGBMRegressor:
    """Return the fixed MVP LightGBM challenger configuration.

    These parameters are a practical research-prototype starting point, not
    Singer-approved policy or a production-approved configuration.
    """
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )
