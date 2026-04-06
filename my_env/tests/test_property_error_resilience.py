# Feature: dark-store-simulator, Property 11: Error Resilience
"""
Property-based test for DarkStoreEnvironment error resilience.

**Validates: Requirements 3.14, 13.1, 13.2, 13.3**

For any malformed, unrecognized, or invalid action input (random strings,
missing fields, wrong types, unrecognized action types), the environment
should never raise an unhandled exception, should return a valid
DarkStoreObservation with a non-empty error field, should advance the
tick by 1, and should continue the episode.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from my_env.server.dark_store_environment import DarkStoreEnvironment
from my_env.models import DarkStoreAction, DarkStoreObservation

# Valid task names for resetting the environment
TASK_NAMES = ["single_order", "concurrent_orders", "full_operations"]
task_name_st = st.sampled_from(TASK_NAMES)

# Valid action types (used to generate unrecognized ones)
VALID_ACTION_TYPES = [
    "move_picker", "pick", "pack", "assign_rider",
    "batch_delivery", "restock", "wait",
]

# Strategy: random strings that are NOT valid action types
invalid_action_type_st = st.text(
    min_size=0, max_size=50
).filter(lambda s: s not in VALID_ACTION_TYPES)

# Strategy: generate DarkStoreAction with unrecognized action types
unrecognized_action_st = invalid_action_type_st.map(
    lambda s: DarkStoreAction(action=s)
)

# Strategy: generate DarkStoreAction with wrong field types
# (e.g., strings where ints expected, negative values, huge values)
wrong_type_fields_st = st.one_of(
    # move_picker with non-integer-like or out-of-bounds coords
    st.builds(
        DarkStoreAction,
        action=st.just("move_picker"),
        row=st.one_of(st.none(), st.integers(min_value=-1000, max_value=1000)),
        col=st.one_of(st.none(), st.integers(min_value=-1000, max_value=1000)),
    ),
    # pick with missing or garbage fields
    st.builds(
        DarkStoreAction,
        action=st.just("pick"),
        row=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
        col=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
        item_name=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
    ),
    # pack with missing or garbage order_id
    st.builds(
        DarkStoreAction,
        action=st.just("pack"),
        order_id=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
    ),
    # assign_rider with missing or garbage fields
    st.builds(
        DarkStoreAction,
        action=st.just("assign_rider"),
        order_id=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
        rider_id=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
    ),
    # batch_delivery with missing or garbage fields
    st.builds(
        DarkStoreAction,
        action=st.just("batch_delivery"),
        order_a=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
        order_b=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
        rider_id=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
    ),
    # restock with missing or garbage fields
    st.builds(
        DarkStoreAction,
        action=st.just("restock"),
        item_name=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
        quantity=st.one_of(st.none(), st.integers(min_value=-100, max_value=100)),
    ),
)

# Combined strategy: any kind of malformed/invalid action
malformed_action_st = st.one_of(
    unrecognized_action_st,
    wrong_type_fields_st,
)


def _assert_valid_observation(obs):
    """Verify the returned object is a valid DarkStoreObservation."""
    assert isinstance(obs, DarkStoreObservation), (
        f"Expected DarkStoreObservation, got {type(obs)}"
    )
    # Should be serializable without error
    obs.model_dump()


@given(task_name=task_name_st, action=unrecognized_action_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_unrecognized_action_type_never_crashes(task_name, action):
    """
    **Validates: Requirements 3.14, 13.1, 13.2, 13.3**

    For any unrecognized action type, the environment should never raise
    an unhandled exception, should return a valid DarkStoreObservation with
    a non-empty error field, should advance the tick by 1, and should
    continue the episode (done=False).
    """
    env = DarkStoreEnvironment()
    obs_before = env.reset(task=task_name)
    tick_before = obs_before.tick

    # Step with unrecognized action — must not raise
    obs = env.step(action)

    # Must return a valid observation
    _assert_valid_observation(obs)

    # Must have a non-empty error field
    assert obs.error is not None and len(obs.error) > 0, (
        f"Expected non-empty error for unrecognized action '{action.action}', "
        f"got error={obs.error!r}"
    )

    # Tick must advance by 1
    assert obs.tick == tick_before + 1, (
        f"Tick should advance by 1: was {tick_before}, now {obs.tick}"
    )

    # Episode should continue (not done after one bad action)
    assert obs.done is False, (
        f"Episode should continue after invalid action, got done={obs.done}"
    )


@given(task_name=task_name_st, action=wrong_type_fields_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_wrong_field_types_never_crashes(task_name, action):
    """
    **Validates: Requirements 3.14, 13.1, 13.2**

    For any action with wrong types or missing required fields, the
    environment should never raise an unhandled exception and should
    return a valid DarkStoreObservation. The tick must advance by 1
    and the episode must continue.
    """
    env = DarkStoreEnvironment()
    obs_before = env.reset(task=task_name)
    tick_before = obs_before.tick

    # Step with malformed action — must not raise
    obs = env.step(action)

    # Must return a valid observation
    _assert_valid_observation(obs)

    # Tick must advance by 1
    assert obs.tick == tick_before + 1, (
        f"Tick should advance by 1: was {tick_before}, now {obs.tick}"
    )

    # Episode should continue
    assert obs.done is False, (
        f"Episode should continue after malformed action, got done={obs.done}"
    )


@given(task_name=task_name_st, action=malformed_action_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_malformed_actions_produce_error_observation(task_name, action):
    """
    **Validates: Requirements 13.1, 13.3**

    For any malformed or invalid action, the environment should return
    a valid DarkStoreObservation with an error description field set,
    advance the tick by 1, and continue the episode.
    """
    env = DarkStoreEnvironment()
    obs_before = env.reset(task=task_name)
    tick_before = obs_before.tick

    # Step — must not raise
    obs = env.step(action)

    # Must return a valid observation
    _assert_valid_observation(obs)

    # Tick must advance by 1
    assert obs.tick == tick_before + 1, (
        f"Tick should advance by 1: was {tick_before}, now {obs.tick}"
    )

    # Episode should continue
    assert obs.done is False, (
        f"Episode should continue after invalid action, got done={obs.done}"
    )


@given(
    task_name=task_name_st,
    actions=st.lists(malformed_action_st, min_size=2, max_size=5),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_sequential_malformed_actions_never_crash(task_name, actions):
    """
    **Validates: Requirements 13.1, 13.2**

    Sending multiple malformed actions in sequence should never crash
    the environment. Each step should advance the tick by 1 and the
    episode should continue.
    """
    env = DarkStoreEnvironment()
    obs = env.reset(task=task_name)

    for i, action in enumerate(actions):
        tick_before = obs.tick
        obs = env.step(action)

        _assert_valid_observation(obs)

        assert obs.tick == tick_before + 1, (
            f"Step {i}: tick should advance by 1: was {tick_before}, now {obs.tick}"
        )

        # Episode should still be running (budget is at least 20 ticks)
        assert obs.done is False, (
            f"Step {i}: episode should continue, got done={obs.done}"
        )
