"""Compare the circular-fracture fork in fork.py with a Darcy resistance model.

Resistance integrates the varying chord width of each circular fracture.
The model still assumes equipotential cross-sections, so comparisons allow 5%
approximation error; conservation and head boundary checks use separate,
stricter tolerances. No plotting or external data is required.

These are acceptance tests, not snapshots of the current solver output.
Differences exceeding the tolerances deliberately fail rather than being
marked as expected failures.
"""

import numpy as np
import pytest

from andfn import DFN, ConstantHeadLine, Fracture, Intersection
from andfn.geometry_functions import map_3d_to_2d

VISCOSITY = 1.002e-3
APERTURE = 1e-2
PERMEABILITY = 2.5e-13
DENSITY = 1000.0
GRAVITY = 9.81
INLET_PRESSURE = 2e6
OUTLET_PRESSURE = 1e6
ANALYTICAL_RTOL = 0.05
HEAD_ATOL = 1e-3
FLOW_RTOL = 1e-6


@pytest.fixture(scope="module", params=[1.0, 2.0], ids=["symmetric", "asymmetric"])
def fork_solution(request):
    lower_permeability_factor = request.param
    transmissivity = PERMEABILITY * APERTURE * DENSITY * GRAVITY / VISCOSITY
    ncoef = 200
    nint = 2 * ncoef
    specifications = [
        ("inlet", 0.9, [-0.45, 0, 0], [0, 0, 1], 1.0),
        ("junction", np.sqrt(0.9**2 - 0.45**2), [0, 0, 0], [1, 0, 0], 1.0),
        ("upper", 0.9, [0.45, 0, 0.2], [0, 0, 1], 1.0),
        ("lower", 0.9, [0.45, 0, -0.2], [0, 0, 1], lower_permeability_factor),
    ]
    fractures = [
        Fracture(
            label,
            transmissivity * factor,
            radius=radius,
            center=np.array(center, dtype=float),
            normal=np.array(normal, dtype=float),
            aperture=APERTURE,
            ncoef=ncoef,
            nint=nint,
        )
        for label, radius, center, normal, factor in specifications
    ]
    dfn = DFN("Analytical fork", discharge_int=100)
    dfn.set_kwargs(
        SE_FACTOR=1.0,
        RHO=DENSITY,
        G=GRAVITY,
        COEF_RATIO=1e-6,
        MAX_ERROR=1e-14,
        MAX_NCOEF=ncoef,
        MAX_ITERATIONS=200,
        DAMPING=0.5,
    )
    dfn.add_fracture(fractures)
    dfn.get_fracture_intersections(ncoef, nint)
    for x, pressure in [(-0.9, INLET_PRESSURE), (0.9, OUTLET_PRESSURE)]:
        dfn.set_constant_head_boundary(
            np.array([x, 0, 0]),
            np.array([1, 0, 0]),
            200,
            pressure / (DENSITY * GRAVITY),
            ncoef=ncoef,
            nint=nint,
        )
    dfn.get_elements()
    assert len([e for e in dfn.elements if isinstance(e, Intersection)]) == 3
    assert len([e for e in dfn.elements if isinstance(e, ConstantHeadLine)]) == 3
    dfn.solve(unconsolidate=True)
    return dfn, lower_permeability_factor


def _reference(lower_permeability_factor):
    # Darcy resistance is mu/(a*k) * integral(ds/width(s)).
    # A disk's width is 2*sqrt(radius**2-s**2), with integral asin(s/radius)/2.
    resistance_scale = VISCOSITY / (APERTURE * PERMEABILITY)
    arm_integral = np.arcsin(0.45 / 0.9)
    junction_integral = 0.5 * np.arcsin(0.2 / np.sqrt(0.9**2 - 0.45**2))
    inlet_resistance = arm_integral * resistance_scale
    upper_resistance = (junction_integral + arm_integral) * resistance_scale
    lower_resistance = (
        junction_integral + arm_integral / lower_permeability_factor
    ) * resistance_scale
    parallel_resistance = 1 / (1 / upper_resistance + 1 / lower_resistance)
    inlet_flow = (INLET_PRESSURE - OUTLET_PRESSURE) / (
        inlet_resistance + parallel_resistance
    )
    junction_pressure = INLET_PRESSURE - inlet_flow * inlet_resistance
    upper_flow = (junction_pressure - OUTLET_PRESSURE) / upper_resistance
    lower_flow = (junction_pressure - OUTLET_PRESSURE) / lower_resistance
    return inlet_flow, upper_flow, lower_flow, junction_pressure


def _boundary(dfn, label):
    elements = [
        e
        for e in dfn.elements
        if isinstance(e, ConstantHeadLine) and e.frac0.label == label
    ]
    assert len(elements) == 1
    return elements[0]


def _intersection(dfn, label):
    elements = [
        e
        for e in dfn.elements
        if isinstance(e, Intersection) and label in (e.frac0.label, e.frac1.label)
    ]
    assert len(elements) == 1
    return elements[0]


def _sample_heads(dfn, fracture, endpoints):
    # Avoid singular endpoints; use points independent of solver quadrature.
    fractions = np.linspace(0.05, 0.95, 31)
    points = endpoints[0] + fractions * (endpoints[1] - endpoints[0])
    return np.array([dfn.calc_head(z, fracture) for z in points])


def test_fork_prescribed_heads(fork_solution):
    dfn, _ = fork_solution
    for label, pressure in [
        ("inlet", INLET_PRESSURE),
        ("upper", OUTLET_PRESSURE),
        ("lower", OUTLET_PRESSURE),
    ]:
        element = _boundary(dfn, label)
        expected_head = pressure / (DENSITY * GRAVITY)
        assert element.head == pytest.approx(expected_head)
        heads = _sample_heads(dfn, element.frac0, element.endpoints0)
        np.testing.assert_allclose(heads, expected_head, rtol=0, atol=HEAD_ATOL)


def test_fork_intersection_head_continuity(fork_solution):
    dfn, _ = fork_solution
    for label in ("inlet", "upper", "lower"):
        element = _intersection(dfn, label)
        heads0 = _sample_heads(dfn, element.frac0, element.endpoints0)
        heads1 = _sample_heads(dfn, element.frac1, element.endpoints1)
        np.testing.assert_allclose(heads0, heads1, rtol=0, atol=HEAD_ATOL)


def test_fork_flow_direction_and_conservation(fork_solution):
    dfn, _ = fork_solution
    inlet = _boundary(dfn, "inlet")
    upper = _boundary(dfn, "upper")
    lower = _boundary(dfn, "lower")
    assert inlet.q < 0
    assert upper.q > 0
    assert lower.q > 0
    assert -inlet.q == pytest.approx(upper.q + lower.q, rel=FLOW_RTOL, abs=1e-18)

    for fracture in dfn.fractures:
        flows = [
            -e.q if isinstance(e, Intersection) and e.frac1 is fracture else e.q
            for e in fracture.elements
            if isinstance(e, (ConstantHeadLine, Intersection))
        ]
        assert abs(sum(flows)) <= FLOW_RTOL * sum(abs(q) for q in flows) + 1e-18


@pytest.mark.parametrize("label", ["inlet", "upper", "lower"])
def test_fork_discharge_matches_darcy_reference(fork_solution, label):
    dfn, factor = fork_solution
    expected_inlet, expected_upper, expected_lower, _ = _reference(factor)
    expected = {
        "inlet": expected_inlet,
        "upper": expected_upper,
        "lower": expected_lower,
    }[label]
    element = _boundary(dfn, label)
    assert abs(element.q) == pytest.approx(expected, rel=ANALYTICAL_RTOL, abs=0)


@pytest.mark.parametrize("label", ["upper", "lower"])
def test_fork_flow_split_matches_darcy_reference(fork_solution, label):
    dfn, factor = fork_solution
    expected_inlet, expected_upper, expected_lower, _ = _reference(factor)
    expected = {"upper": expected_upper, "lower": expected_lower}[label]
    inlet_flow = -_boundary(dfn, "inlet").q
    flow_fraction = _boundary(dfn, label).q / inlet_flow
    assert flow_fraction == pytest.approx(
        expected / expected_inlet, rel=ANALYTICAL_RTOL, abs=0
    )


def test_fork_branch_response_to_permeability(fork_solution):
    dfn, factor = fork_solution
    upper_flow = _boundary(dfn, "upper").q
    lower_flow = _boundary(dfn, "lower").q
    if factor == 1.0:
        # Reflection symmetry is exact even for circular fractures.
        assert upper_flow == pytest.approx(lower_flow, rel=FLOW_RTOL, abs=1e-18)
    else:
        assert lower_flow > upper_flow, (
            "Increasing the lower branch permeability must increase its flow "
            "relative to the otherwise identical upper branch."
        )


def test_fork_junction_head_matches_darcy_reference(fork_solution):
    dfn, factor = fork_solution
    _, _, _, expected_pressure = _reference(factor)
    junction = next(f for f in dfn.fractures if f.label == "junction")
    z = map_3d_to_2d(np.array([0.0, 0.0, 0.0]), junction)
    actual_head = dfn.calc_head(z, junction)
    expected_head = expected_pressure / (DENSITY * GRAVITY)
    head_drop = (INLET_PRESSURE - OUTLET_PRESSURE) / (DENSITY * GRAVITY)
    assert actual_head == pytest.approx(
        expected_head, rel=0, abs=ANALYTICAL_RTOL * head_drop
    )
