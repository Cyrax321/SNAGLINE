from snagline.dedup import EventDeduplicator


def test_dedup_duplicate_detection():
    dedup = EventDeduplicator()
    payload = {"key": "val", "count": 1}
    assert not dedup.is_duplicate(payload)
    assert dedup.is_duplicate(payload)
    assert not dedup.is_duplicate({"key": "val", "count": 2})
