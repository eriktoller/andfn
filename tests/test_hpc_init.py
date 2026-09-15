from andfn.hpc import CACHE, PARALLEL


def test_hpc_flags_are_booleans():
    assert isinstance(PARALLEL, bool)
    assert isinstance(CACHE, bool)
