from __future__ import annotations

import hmac

import bcrypt


def hash_api_key(api_key: str) -> str:
    if not api_key:
        raise ValueError("API key cannot be empty.")
    return bcrypt.hashpw(api_key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_api_key_hash(candidate: str, stored_hash: str) -> bool:
    if not candidate or not stored_hash:
        return False
    try:
        return bcrypt.checkpw(candidate.encode("utf-8"), stored_hash.encode("utf-8"))
    except ValueError:
        return False


def constant_time_equals(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
