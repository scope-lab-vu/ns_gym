"""The parameter tables on the docs pages match what the wrappers accept.

Each environment page lists its tunable parameters in a Markdown table whose
first column is the parameter name in backticks. A name on the page that the
wrapper rejects sends users into an ``AssertionError``; a name the wrapper
accepts that is missing from the page is undocumented.
"""

import re
from pathlib import Path

import pytest

from ns_gym.base import TUNABLE_PARAMS

ENV_PAGES = Path(__file__).resolve().parents[1] / "docs" / "source" / "env_pages"

pytestmark = pytest.mark.skipif(
    not ENV_PAGES.is_dir(), reason="docs sources are not available (not a source checkout)"
)

#: docs page -> unwrapped env class names it documents
PAGES = {
    "classic_control/cartpole.md": ["CartPoleEnv"],
    "classic_control/acrobot.md": ["AcrobotEnv"],
    "classic_control/pendulum.md": ["PendulumEnv"],
    "classic_control/mountaincar.md": ["MountainCarEnv", "Continuous_MountainCarEnv"],
    "toytext/frozenlake.md": ["FrozenLakeEnv"],
    "toytext/cliffwalking.md": ["CliffWalkingEnv"],
    "toytext/bridge.md": ["Bridge"],
    "mujoco/antenv.md": ["AntEnv"],
    "mujoco/halfcheetah.md": ["HalfCheetahEnv"],
    "mujoco/hopper.md": ["HopperEnv"],
    "mujoco/humanoid.md": ["HumanoidEnv"],
    "mujoco/humanoid_standup.md": ["HumanoidStandupEnv"],
    "mujoco/inverted_pendulum.md": ["InvertedPendulumEnv"],
    "mujoco/inverted_double_pendulum.md": ["InvertedDoublePendulumEnv"],
    "mujoco/reacher.md": ["ReacherEnv"],
    "mujoco/swimmer.md": ["SwimmerEnv"],
    "mujoco/pusher.md": ["PusherEnv"],
}

_PARAM_ROW = re.compile(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|", flags=re.MULTILINE)


def documented_parameters(page: str) -> set:
    return set(_PARAM_ROW.findall((ENV_PAGES / page).read_text()))


@pytest.mark.parametrize("page", PAGES)
def test_documented_parameters_match_accepted_parameters(page):
    class_names = PAGES[page]
    accepted = set().union(*(TUNABLE_PARAMS[name] for name in class_names))
    documented = documented_parameters(page)

    assert documented, f"no parameter table found on {page}"
    assert documented == accepted, (
        f"{page}: documented but rejected {sorted(documented - accepted)}; "
        f"accepted but undocumented {sorted(accepted - documented)}"
    )


def test_every_environment_page_is_checked():
    """A new environment page must be added to ``PAGES`` above."""
    on_disk = {
        str(path.relative_to(ENV_PAGES))
        for path in ENV_PAGES.glob("*/*.md")
        if path.name != "index.md" and path.parent.name != "nsgym_custom"
    }
    assert on_disk == set(PAGES)
