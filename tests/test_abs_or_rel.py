import numcodecs
import numcodecs.registry
import numpy as np
import pytest

INNER = dict(id="eb_quantize", eb="$eb_abs", codec=dict(id="zlib", level=1))


def test_from_config():
    codec = numcodecs.registry.get_codec(
        dict(id="abs_or_rel", eb_abs=0.01, eb_rel=0.05, codec=INNER)
    )
    assert codec.__class__.__name__ == "AbsOrRelErrorBoundedCodec"
    assert codec.__class__.__module__ == "numcodecs_abs_or_rel"
    config = codec.get_config()
    assert config == dict(
        id="abs_or_rel", eb_abs=0.01, eb_rel=0.05, codec=INNER, eb_abs_marker="$eb_abs"
    )
    assert numcodecs.registry.get_codec(config).get_config() == config


def test_invalid():
    from numcodecs_abs_or_rel import AbsOrRelErrorBoundedCodec

    with pytest.raises(ValueError):
        AbsOrRelErrorBoundedCodec(eb_abs=0.0, eb_rel=0.1, codec=INNER)
    with pytest.raises(ValueError):
        AbsOrRelErrorBoundedCodec(eb_abs=0.1, eb_rel=np.inf, codec=INNER)
    with pytest.raises(TypeError):
        AbsOrRelErrorBoundedCodec(eb_abs=0.1, eb_rel=0.1, codec="zlib")
    with pytest.raises(TypeError):
        AbsOrRelErrorBoundedCodec(eb_abs=0.1, eb_rel=0.1, codec=INNER).encode(
            np.arange(3)
        )


def test_map():
    from numcodecs_abs_or_rel import AbsOrRelErrorBoundedCodec

    codec = AbsOrRelErrorBoundedCodec(eb_abs=0.1, eb_rel=0.1, codec=INNER)
    mapped = codec.map(lambda c: c)
    config = mapped.get_config()
    assert config["codec"]["id"] == "eb_quantize" and config["codec"]["eb"] == "$eb_abs"
    data = np.linspace(-5.0, 5.0, 101)
    np.testing.assert_array_equal(
        np.asarray(mapped.decode(mapped.encode(data))),
        np.asarray(codec.decode(codec.encode(data))),
    )


def check_roundtrip(data: np.ndarray, eb_abs: float, eb_rel: float):
    codec = numcodecs.registry.get_codec(
        dict(id="abs_or_rel", eb_abs=eb_abs, eb_rel=eb_rel, codec=INNER)
    )

    encoded = codec.encode(data)
    decoded = np.asarray(codec.decode(encoded))

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape

    finite = np.isfinite(data)
    x = data[finite].astype(np.float64)
    error = np.abs(decoded[finite].astype(np.float64) - x)
    assert np.all(error <= np.maximum(eb_abs, eb_rel * np.abs(x)))
    # signs are preserved in the relative regime
    large = np.abs(x) > eb_abs
    assert np.all(np.sign(decoded[finite][large]) == np.sign(x[large]))

    out = np.empty_like(data)
    codec.decode(encoded, out=out)
    np.testing.assert_array_equal(out, decoded)

    return len(encoded)


def test_roundtrip():
    rng = np.random.default_rng(0)
    # values spanning many orders of magnitude around the transition eb_abs / eb_rel
    data = np.concatenate(
        [
            rng.normal(size=1000) * 10.0 ** rng.integers(-6, 3, size=1000),
            np.zeros(50),
            rng.uniform(-0.3, 0.3, size=200),
        ]
    ).reshape(50, 25)
    for eb_abs, eb_rel in [(0.01, 0.05), (1e-7, 0.1), (1.0, 0.01), (0.05, 0.5)]:
        check_roundtrip(data, eb_abs, eb_rel)
        check_roundtrip(data.astype(np.float32), eb_abs, eb_rel)
    check_roundtrip(np.array([np.nan, 1.0, -2.0, 0.0]), 0.1, 0.1)
    check_roundtrip(np.zeros((0, 3)), 0.1, 0.1)
    check_roundtrip(np.array(3.0), 0.1, 0.1)


def test_transition_region():
    # dense sampling across the linear/logarithmic transition
    eb_abs, eb_rel = 0.01, 0.05
    x0 = eb_abs / eb_rel
    data = np.concatenate(
        [np.linspace(-3 * x0, 3 * x0, 5001), -np.linspace(0.9 * x0, 1.1 * x0, 1001)]
    )
    check_roundtrip(data, eb_abs, eb_rel)


def test_masked():
    from numcodecs_mask import MaskMetaCodec

    rng = np.random.default_rng(1)
    data = rng.normal(size=(30, 40)) * 10.0 ** rng.integers(-4, 2, size=(30, 40))
    mask = rng.random(data.shape) < 0.3
    data[mask] = np.nan
    codec = MaskMetaCodec(
        mask=np.nan,
        codec=dict(id="abs_or_rel", eb_abs=0.01, eb_rel=0.05, codec=INNER),
        bitmap_codec=dict(id="packbits"),
    )
    decoded = np.asarray(codec.decode(codec.encode(data)))
    np.testing.assert_array_equal(np.isnan(decoded), mask)
    x = data[~mask]
    assert np.all(np.abs(decoded[~mask] - x) <= np.maximum(0.01, 0.05 * np.abs(x)))
