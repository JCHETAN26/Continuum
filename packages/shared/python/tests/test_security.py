from continuum_shared.security import constant_time_equals, hash_api_key, verify_api_key_hash


def test_api_key_hash_round_trip() -> None:
    stored_hash = hash_api_key("secret-value")

    assert verify_api_key_hash("secret-value", stored_hash)
    assert not verify_api_key_hash("wrong-value", stored_hash)


def test_constant_time_equals() -> None:
    assert constant_time_equals("same", "same")
    assert not constant_time_equals("same", "different")
