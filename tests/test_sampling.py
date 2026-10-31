from snagline.sampling import AdaptiveSampler


def test_sampler_boundary_always_never():
    always = AdaptiveSampler(1.0)
    never = AdaptiveSampler(0.0)
    assert all(always.should_sample() for _ in range(10))
    assert not any(never.should_sample() for _ in range(10))
