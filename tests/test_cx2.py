import pathlib

import numpy as np
import pytest
import yaml

from bosonoise.channels.cx2 import BathModel, LadderGrid, calibrate_cx2, initial_blocks, ladder_at, simulate_cx2
from bosonoise.dynamics.ule import detailed_balance, flat
from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def grid():
    ref = yaml.safe_load(open(ROOT / "data" / "calibration" / "ocelot_A2S3_fitA.yaml", encoding="utf-8"))
    m = StorageCouplerAncilla(CellParams(**ref["params"]), Truncation(n_s=10, n_c=4, n_q=5, ncut=15))
    return LadderGrid([ladder_at(m, f, 8) for f in (0.0, 0.3, 0.45, 0.5)])


def test_total_parity_selection_rule(grid):
    """Charge operators are odd under the total parity (-1)^(k + c + n): the photon-assisted
    |g~,n> -> |e~,n-1> element vanishes, the direct |f~,n> -> |g~,n+1> one does not; the
    (parity-even) ancilla number operator does drive |g~,n> -> |e~,n-1>."""
    on = grid.ladders[-1]
    for n in range(1, 8):
        assert abs(on.X["q"][1, n - 1, 0, n]) < 1e-10
        assert abs(on.X["c"][1, n - 1, 0, n]) < 1e-10
    assert abs(on.X["q"][0, 2, 2, 1]) > 1e-3
    assert abs(on.X["Nq"][1, 0, 0, 1]) > 1e-3


def test_phase_convention_keeps_ladder_coherent(grid):
    for lad in grid.ladders:
        x = np.array([lad.X["q"][1, n, 2, n] for n in range(8)])  # f -> e, same photon number
        assert np.all(x > 0) or np.all(x < 0)


def test_cx2_turn_and_decay_bookkeeping(grid):
    pulse = calibrate_cx2(grid, ramp=32.0)
    t = np.arange(0.0, pulse.duration, 0.05)
    assert abs(np.trapezoid([abs(grid.chi_gf(f)) for f in pulse.flux(t)], t) - 1) < 2e-3
    G = 1 / 20e3  # f -> e rate, 1/ns
    idle = grid.ladders[0]
    bath = BathModel({"q": detailed_balance(flat(G / idle.X["q"][1, 0, 2, 0] ** 2), None)}, standard=True)
    blk = simulate_cx2(grid, pulse, bath, initial_blocks(1.0, 2, 7), 7, blocks=True)
    pops = np.trace(blk, axis1=1, axis2=2).real
    assert abs(pops.sum() - 1) < 1e-6
    expected = 1 - np.exp(-pulse.duration * G)  # undressed f population decays at G throughout
    assert abs(pops[1] + pops[0] - expected) < 1e-3 * expected


def test_stabilizer_fault_tables_close_on_the_full_table(grid):
    """none + in_a + in_b = the full stabilizer table up to two-fault terms; f->e in CX(a) has probability
    (1/2) G T_a (f branch only) and is heralded, not converted (chi-matching is not assumed here)."""
    from bosonoise.channels.conversion import AsymptoticProjector
    from bosonoise.channels.cx2 import calibrate_phases
    from bosonoise.channels.cycle import CXCell, fault_tables, stabilizer_table, summarize_fault

    pulse = calibrate_cx2(grid, ramp=32.0, turns=0.5)
    projs = {1.0: AsymptoticProjector(1.0), 2.0: AsymptoticProjector(2.0)}
    corr = calibrate_phases(grid, pulse, 7, 1.0, projs[1.0], f_target_flipped=True)
    G = 1 / 20e3
    idle = grid.ladders[0]
    bath = BathModel({"q": detailed_balance(flat(G / idle.X["q"][1, 0, 2, 0] ** 2), None)}, standard=True)
    cell = CXCell(grid, pulse, corr, bath, projs, nmax=7)
    full = stabilizer_table(cell, cell, 2.0)
    parts = fault_tables(cell, cell, 2.0, "f->e")
    for B in ("g", "f"):
        diff = full[B] - (parts["none"][B] + parts["in_a"][B] + parts["in_b"][B])
        assert np.abs(diff).sum() < 1e-3  # two f->e faults (one in each gate) are O((G T)^2)
    sa = summarize_fault(parts["in_a"])
    assert abs(sa["p_fault"] - 0.5 * (1 - np.exp(-pulse.duration * G))) < 0.03 * sa["p_fault"]
    assert sa["herald"] > 0.5  # most single f->e faults end in e (erasure flag)


def test_component_tables_separate_sequences(grid):
    """Multi-class split (f->e, e->g): the components sum to the full table, and the double decay f->e->g is
    a separate component that converts into cat errors much more often than f->e alone."""
    from bosonoise.channels.conversion import AsymptoticProjector
    from bosonoise.channels.cx2 import calibrate_phases
    from bosonoise.channels.cycle import CXCell, component_tables, stabilizer_table, summarize_fault

    pulse = calibrate_cx2(grid, ramp=32.0, turns=0.5)
    projs = {1.0: AsymptoticProjector(1.0), 2.0: AsymptoticProjector(2.0)}
    corr = calibrate_phases(grid, pulse, 7, 1.0, projs[1.0], f_target_flipped=True)
    G = 1 / 5e3  # fast decay so that the double decay is resolved
    idle = grid.ladders[0]
    bath = BathModel({"q": detailed_balance(flat(G / idle.X["q"][1, 0, 2, 0] ** 2), None)}, standard=True)
    cell = CXCell(grid, pulse, corr, bath, projs, nmax=7)
    full = stabilizer_table(cell, cell, 2.0)
    comps = component_tables(cell, cell, 2.0, ("f->e", "e->g"))
    assert set(comps) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    for B in ("g", "f"):
        assert np.abs(full[B] - sum(comps[c][B] for c in comps)).sum() < 2e-3
    dd, fe = summarize_fault(comps[(1, 1)]), summarize_fault(comps[(1, 0)])
    assert dd["p_fault"] > 0 and dd["odd"] > 5 * fe["odd"]


def test_herald_tables_reduce_to_component_tables(grid):
    """Summing the buffer-click resolution over clicks gives the plain component tables; clicks vanish as eta -> 0."""
    from bosonoise.channels.conversion import AsymptoticProjector
    from bosonoise.channels.cx2 import calibrate_phases
    from bosonoise.channels.cycle import CXCell, component_tables, herald_component_tables, summarize_herald

    pulse = calibrate_cx2(grid, ramp=32.0, turns=0.5)
    projs = {1.0: AsymptoticProjector(1.0), 2.0: AsymptoticProjector(2.0)}
    corr = calibrate_phases(grid, pulse, 7, 1.0, projs[1.0], f_target_flipped=True)
    G = 1 / 5e3
    idle = grid.ladders[0]
    bath = BathModel({"q": detailed_balance(flat(G / idle.X["q"][1, 0, 2, 0] ** 2), None)}, standard=True)
    cell = CXCell(grid, pulse, corr, bath, projs, nmax=7)
    plain = component_tables(cell, cell, 2.0, ("f->e", "e->g"))
    for eta in (1e-7, 0.5):
        her = herald_component_tables(cell, cell, 2.0, ("f->e", "e->g"), eta)
        for c in plain:
            for B in ("g", "f"):
                assert np.abs(her[c][B].sum(axis=(1, 3)) - plain[c][B]).max() < 1e-9
        s = summarize_herald(her[(1, 1)])  # the double decay f->e->g
        if eta < 1e-3:
            assert s["x_b_heralded"] < 1e-5 and s["false_click_b"] < 1e-5
        else:
            # a decay during a gate leaves a partial rotation, which relaxes with a buffer burst (heralded); a gate
            # skipped entirely (the ancilla reached g before CX(b)) leaves cat b in the code space: a silent error
            assert s["x_a_heralded"] / s["x_a"] > s["x_b_heralded"] / s["x_b"]
            assert s["odd_silent"] > 0.3 * s["odd"]


def test_fault_injection_splits_the_full_evolution(grid):
    """First-order Dyson split: with f->e as the selected family, a second f->e jump is impossible, so
    rho_0 + rho_1 equals the full result and Tr rho_1 is the f->e probability over the pulse."""
    pulse = calibrate_cx2(grid, ramp=32.0)
    G = 1 / 20e3
    idle = grid.ladders[0]
    bath = BathModel({"q": detailed_balance(flat(G / idle.X["q"][1, 0, 2, 0] ** 2), None)}, standard=True)
    sig0 = initial_blocks(1.0, 2, 7)
    full = simulate_cx2(grid, pulse, bath, sig0, 7, blocks=True)
    r0, r1 = simulate_cx2(grid, pulse, bath, sig0, 7, blocks=True, fault=lambda k: k[1:3] == (2, 1))
    assert np.abs(r0 + r1 - full).max() < 1e-12
    p1 = np.trace(r1, axis1=1, axis2=2).real
    assert abs(p1.sum() - (1 - np.exp(-pulse.duration * G))) < 2e-3 * p1.sum()
    assert p1[2] == 0.0  # the one-jump part has left f


def test_coupler_excited_branches():
    ref = yaml.safe_load(open(ROOT / "data" / "calibration" / "ocelot_A2S3_fitA.yaml", encoding="utf-8"))
    m = StorageCouplerAncilla(CellParams(**ref["params"]), Truncation(n_s=9, n_c=5, n_q=5, ncut=15))
    from bosonoise.channels.cx2 import WITH_COUPLER

    grid = LadderGrid([ladder_at(m, f, 6, branches=WITH_COUPLER) for f in (0.0, 0.3, 0.5)])
    assert grid.n_branches == 6
    on = grid.ladders[-1]
    # the coupler excitation costs roughly the coupler frequency and shifts the storage (chi_sc < 0)
    assert 5.5 < on.E[3, 0] - on.E[0, 0] < 7.5
    assert grid.chi_rel(0.5, 3) < 0
    # coupler decay |1_c g, n> -> |0_c g, n> is a charge (n_c) transition of order the bare element
    assert abs(on.X["c"][0, 2, 3, 2]) > 0.5 * on.nc01
    # the undressed (standard) model never uses the coupler-excited branches
    bath = BathModel({"q": detailed_balance(flat(1e-5), 50.0)}, standard=True)
    assert all(max(i, j) <= 2 for (_, i, j, _) in bath.families(grid, 5))


def test_stabilizer_fault_table_bookkeeping(grid):
    """Two chained CX (half turns) on the same test cell: normalization, heralds from f->e, no-bath limit."""
    from bosonoise.channels.conversion import AsymptoticProjector
    from bosonoise.channels.cycle import CXCell, stabilizer_table, summarize
    from bosonoise.channels.cx2 import calibrate_phases

    pulse = calibrate_cx2(grid, ramp=32.0, turns=0.5)
    projs = {1.0: AsymptoticProjector(1.0), 2.0: AsymptoticProjector(2.0)}
    corr = calibrate_phases(grid, pulse, 7, 1.0, projs[1.0], f_target_flipped=True)
    nob = CXCell(grid, pulse, corr, BathModel({}), projs, nmax=7)
    s0 = summarize(stabilizer_table(nob, nob, 2.0))
    assert s0["p_herald"] < 1e-12 and s0["p_silent_syndrome"] < 1e-12
    G = 1 / 20e3  # f -> e rate (1/ns), undressed
    idle = grid.ladders[0]
    bath = BathModel({"q": detailed_balance(flat(G / idle.X["q"][1, 0, 2, 0] ** 2), None)}, standard=True)
    cell = CXCell(grid, pulse, corr, bath, projs, nmax=7)
    T = stabilizer_table(cell, cell, 2.0)
    assert abs(T["g"].sum() - 1) < 1e-9 and abs(T["f"].sum() - 1) < 1e-9
    s = summarize(T)
    # heralds: f-branch (weight 1/2) decays f->e during two gates; e->g (rate G/2 here) removes a little
    expected = 0.5 * (1 - np.exp(-2 * pulse.duration * G))
    assert abs(s["p_herald"] - expected) < 0.05 * expected
    assert s["p_xx"] <= min(s["p_x_a"], s["p_x_b"]) + 1e-15
    # logical-bit-flip contribution: odd number of cat X errors (X_a X_b acts trivially on Z_1 ... Z_d)
    from bosonoise.channels.cycle import odd_parity

    assert abs(odd_parity(T) - (s["p_x_a"] + s["p_x_b"] - 2 * s["p_xx"])) < 1e-14
    assert abs(odd_parity(T, "g") - (s["p_x_a_branch_g"] + s["p_x_b_branch_g"] - 2 * s["p_xx_branch_g"])) < 1e-14
