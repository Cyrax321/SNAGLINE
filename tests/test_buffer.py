from snagline.buffer import BoundedEventBuffer


def test_buffer_capacity_eviction():
    buf = BoundedEventBuffer(capacity=2)
    assert not buf.push("a")
    assert not buf.push("b")
    assert buf.push("c")
    assert buf.items == ["b", "c"]
