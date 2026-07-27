import pytest
from continuum_shared.artifacts import (
    build_demo_artifact_manifest,
    decode_manifest,
    encode_manifest,
    parse_s3_uri,
    sha256_hex,
)


def test_manifest_encoding_is_stable():
    manifest = build_demo_artifact_manifest(
        version="2026.07.26-test",
        base_model="continuum/hash-embedding-demo",
        embedding_dim=384,
        metrics={"mrr": 0.72},
        baseline_metrics={"mrr": 0.58},
        improvement_pct=0.24,
    )

    encoded = encode_manifest(manifest)

    assert decode_manifest(encoded) == manifest
    assert sha256_hex(encoded) == sha256_hex(encode_manifest(manifest))


def test_parse_s3_uri():
    assert parse_s3_uri("s3://continuum-models/v1/model.json") == (
        "continuum-models",
        "v1/model.json",
    )

    with pytest.raises(ValueError):
        parse_s3_uri("builtin://continuum/hash-embedding")
