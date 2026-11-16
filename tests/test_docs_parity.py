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

from _helpers import known_bug

ENV_PAGES = Path(__file__).resolve().parents[1] / "docs" / "source" / "env_pages"

pytestmark = pytest.mark.skipif(
    not ENV_PAGES.is_dir(), reason="docs sources are not available (not a source checkout)"
)

#: docs page -> (unwrapped env class names it documents, finding id if it is known to disagree)
PAGES = {
    "classic_control/cartpole.md": (["CartPoleEnv"], None),
    "classic_control/acrobot.md": (["AcrobotEnv"], None),
    "classic_control/pendulum.md": (["PendulumEnv"], None),
    "classic_control/mountaincar.md": (["MountainCarEnv", "Continuous_MountainCarEnv"], None),
    "toytext/frozenlake.md": (["FrozenLakeEnv"], None),
    "toytext/cliffwalking.md": (["CliffWalkingEnv"], None),
    "toytext/bridge.md": (["Bridge"], "D5"),
    "mujoco/antenv.md": (["AntEnv"], "A3"),
    "mujoco/halfcheetah.md": (["HalfCheetahEnv"], None),
    "mujoco/hopper.md": (["HopperEnv"], None),
    "mujoco/humanoid.md": (["HumanoidEnv"], "A3"),
    "mujoco/humanoid_standup.md": (["HumanoidStandupEnv"], "A3"),
    "mujoco/inverted_pendulum.md": (["InvertedPendulumEnv"], "D2"),
    "mujoco/inverted_double_pendulum.md": (["InvertedDoublePendulumEnv"], "D2"),
    "mujoco/reacher.md": (["ReacherEnv"], "D2"),
    "mujoco/swimmer.md": (["SwimmerEnv"], "D2"),
    "mujoco/pusher.md": (["PusherEnv"], None),
}

_PARAM_ROW = re.compile(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|", flags=re.MULTILINE)


def documented_parameters(page: str) -> set:
    return set(_PARAM_ROW.findall((ENV_PAGES / page).read_text()))


@pytest.mark.parametrize(
    "page",
    [
        pytest.param(page, marks=[known_bug(bug)] if bug else [])
        for page, (_, bug) in PAGES.items()
    ],
)
def test_documented_parameters_match_accepted_parameters(page):
    class_names, _ = PAGES[page]
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
