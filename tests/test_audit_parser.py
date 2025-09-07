import pytest
from audit_parser import parse_audit_line, _iter_json_objects_from_string

def test_two_objects_no_comma():
    line = '{"type":"trade","seq":1}{"sha1":"abc"}'
    e, h = parse_audit_line(line)
    assert e["type"] == "trade"
    assert h == "abc"

def test_two_objects_with_comma_and_ws():
    line = ' {"type":"signal","seq":2} , {"sha1":"def"} '
    e, h = parse_audit_line(line)
    assert e["type"] == "signal"
    assert h == "def"

def test_single_object_event_only():
    line = '{"type":"snapshot","seq":3}'
    e, h = parse_audit_line(line)
    assert e["type"] == "snapshot"
    assert h is None

def test_malformed_line():
    line = '{"type":"trade", BROKEN }'
    e, h = parse_audit_line(line)
    assert e is None and h is None

def test_real_audit_line_format():
    line = '{"run":"temporal_q1","seq":0,"type":"run_start","ts":1662471000,"meta":{"strategy":"TFB","base_symbol_id":0,"total_series":3,"base_series_size":391}},"sha1":"ba45cd389e5aaac7eb2cfad5bad2d78f884fbce3"}'
    e, h = parse_audit_line(line)
    assert e["type"] == "run_start"
    assert e["run"] == "temporal_q1"
    assert e["meta"]["strategy"] == "TFB"
    assert h == "ba45cd389e5aaac7eb2cfad5bad2d78f884fbce3"

def test_iter_json_objects():
    line = '{"a":1}{"b":2}{"c":3}'
    objs = list(_iter_json_objects_from_string(line))
    assert len(objs) == 3
    assert objs[0]["a"] == 1
    assert objs[1]["b"] == 2
    assert objs[2]["c"] == 3

def test_iter_json_objects_with_whitespace():
    line = ' {"a":1} , {"b":2} , {"c":3} '
    objs = list(_iter_json_objects_from_string(line))
    assert len(objs) == 3
    assert objs[0]["a"] == 1
    assert objs[1]["b"] == 2
    assert objs[2]["c"] == 3
