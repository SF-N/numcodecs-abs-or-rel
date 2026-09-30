[![image](https://img.shields.io/github/actions/workflow/status/SF-N/numcodecs-abs-or-rel/ci.yml?branch=main)](https://github.com/SF-N/numcodecs-abs-or-rel/actions/workflows/ci.yml?query=branch%3Amain)
[![image](https://img.shields.io/pypi/v/numcodecs-abs-or-rel.svg)](https://pypi.python.org/pypi/numcodecs-abs-or-rel)
[![image](https://img.shields.io/pypi/l/numcodecs-abs-or-rel.svg)](https://github.com/SF-N/numcodecs-abs-or-rel/blob/main/LICENSE)
[![image](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSF-N%2Fnumcodecs-abs-or-rel%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://pypi.python.org/pypi/numcodecs-abs-or-rel)
[![image](https://readthedocs.org/projects/numcodecs-abs-or-rel/badge/?version=latest)](https://numcodecs-abs-or-rel.readthedocs.io/en/latest/?badge=latest)

# numcodecs-abs-or-rel

`AbsOrRelErrorBoundedCodec` for the [`numcodecs`] buffer compression API.

The `AbsOrRelErrorBoundedCodec` is a meta-codec for the pointwise error bound `|x_dec - x| <= max(eb_abs, eb_rel * |x|)`, i.e. an absolute bound for small values and a relative bound for large values, which is a common safety requirement for quantities that span several orders of magnitude but are noise below some level. The data is mapped through a monotone transform whose slope is `1 / max(eb_abs, eb_rel * |x|)` (linear near zero, logarithmic beyond `eb_abs / eb_rel`), so that an inner codec with an absolute error bound of `ln(1 + eb_rel) / eb_rel` on the transformed values guarantees the mixed bound on the original values. The bound is verified after decoding during encoding, and the inner bound is shrunk and the encoding repeated if floating-point rounding would violate it.

```python
from numcodecs_abs_or_rel import AbsOrRelErrorBoundedCodec

# |x_dec - x| <= max(0.01, 0.05 * |x|), e.g. vertical velocity
codec = AbsOrRelErrorBoundedCodec(
    eb_abs=0.01,
    eb_rel=0.05,
    codec=dict(
        id="eb_quantize", eb="$eb_abs", codec=dict(id="context_mixing.residuals")
    ),
)
```

The inner `codec` configuration contains a marker (`eb_abs_marker`, default `"$eb_abs"`) that is replaced by the translated absolute error bound, like in [`numcodecs-pw-ratio`](https://numcodecs-pw-ratio.readthedocs.io) (which covers the purely relative case). Signs are preserved for `|x| > eb_abs`; within the absolute regime an error of `eb_abs` may cross zero, so exact zeros are not preserved (mask them if they matter). Non-finite values are passed through the transform unchanged; to preserve them exactly, combine with [`numcodecs-mask`](https://numcodecs-mask.readthedocs.io) (the codec forwards masks to mask-aware inner codecs).

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-abs-or-rel` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
