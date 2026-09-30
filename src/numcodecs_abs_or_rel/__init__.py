"""
[`AbsOrRelErrorBoundedCodec`][numcodecs_abs_or_rel.AbsOrRelErrorBoundedCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["AbsOrRelErrorBoundedCodec"]

import copy
import math
from collections.abc import Callable
from functools import reduce
from io import BytesIO

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
from numcodecs_mask.abc import MaskAwareCodecMixin
from typing_extensions import Buffer  # MSPV 3.12

Mask = np.ndarray[tuple[int, ...], np.dtype[np.bool]]


def _replace_marker(config: object, marker: object, value: object) -> object:
    """Replace values equal to `marker` in a (nested) codec config by `value`."""
    if isinstance(config, dict):
        return {
            k: (
                value
                if (type(v) is type(marker) and v == marker)
                else _replace_marker(v, marker, value)
            )
            for k, v in config.items()
        }
    if isinstance(config, (list, tuple)):
        return type(config)(_replace_marker(v, marker, value) for v in config)
    return config


class AbsOrRelErrorBoundedCodec(Codec, CodecCombinatorMixin, MaskAwareCodecMixin):
    r"""
    Meta-codec that bounds the pointwise error by the maximum of an absolute
    and a relative error bound,

    $$ |\hat{x} - x| \leq \max(\epsilon_{abs}, \epsilon_{rel} \cdot |x|), $$

    by transforming the data and encoding it with an absolute-error-bounded
    `codec`.

    The odd, monotone transform `y = f(x)` has the slope
    `1 / max(eb_abs, eb_rel * |x|)`: it is linear (`x / eb_abs`) for
    `|x| <= eb_abs / eb_rel` and logarithmic beyond. An absolute error of at
    most `ln(1 + eb_rel) / eb_rel` (which is slightly below `1`) on `y`
    therefore guarantees the mixed bound on `x`, for every finite value and
    across the transition between the two regimes. Zero maps to zero and signs
    are preserved for `|x| > eb_abs` (within the absolute regime the error may
    cross zero). The translated absolute error bound replaces the
    `eb_abs_marker` in the `codec` configuration, like in
    [`numcodecs-pw-ratio`](https://numcodecs-pw-ratio.readthedocs.io), which
    covers the purely relative case.

    Because the transform is evaluated in floating-point arithmetic, the bound
    is verified on the reconstructed values during encoding; if it is
    exceeded, the inner bound is shrunk and the encoding repeated.

    Non-finite values are passed through the transform unchanged; combine
    with a masking meta-codec such as
    [`numcodecs_mask.MaskMetaCodec`](https://numcodecs-mask.readthedocs.io)
    to preserve them exactly. This codec implements the
    [`MaskAwareCodecMixin`][numcodecs_mask.abc.MaskAwareCodecMixin]: masked
    values are excluded from the verification and the mask is forwarded to
    the inner `codec` if it is mask-aware as well.

    Parameters
    ----------
    eb_abs : float
        The positive absolute error bound.
    eb_rel : float
        The positive relative error bound.
    codec : dict
        The configuration of the codec that encodes the transformed data with
        an absolute error bound; it must contain the `eb_abs_marker`.
    eb_abs_marker : str, optional
        The marker for the absolute error bound in `codec`.
    """

    __slots__: tuple[str, ...] = ("_eb_abs", "_eb_rel", "_codec", "_eb_abs_marker")
    _eb_abs: float
    _eb_rel: float
    _codec: dict
    _eb_abs_marker: str

    codec_id: str = "abs_or_rel"  # type: ignore

    def __init__(
        self,
        *,
        eb_abs: float,
        eb_rel: float,
        codec: dict,
        eb_abs_marker: str = "$eb_abs",
    ) -> None:
        if not (math.isfinite(eb_abs) and eb_abs > 0):
            raise ValueError("eb_abs must be finite and positive")
        if not (math.isfinite(eb_rel) and eb_rel > 0):
            raise ValueError("eb_rel must be finite and positive")
        if not isinstance(codec, dict):
            raise TypeError(
                "codec must be a configuration dict containing the eb_abs_marker"
            )

        self._eb_abs = float(eb_abs)
        self._eb_rel = float(eb_rel)
        self._codec = copy.deepcopy(codec)
        self._eb_abs_marker = eb_abs_marker

    @property
    def _inner_bound(self) -> float:
        # an error of ln(1 + r) / r in y gives at most a relative error of r
        return math.log1p(self._eb_rel) / self._eb_rel

    def _inner(self, eb_abs: float) -> Codec:
        config = _replace_marker(self._codec, self._eb_abs_marker, float(eb_abs))
        assert isinstance(config, dict)
        return numcodecs.registry.get_codec(config)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        a, r = self._eb_abs, self._eb_rel
        x0 = a / r
        ax = np.abs(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(ax <= x0, ax / a, x0 / a + np.log(np.maximum(ax, x0) / x0) / r)
        return np.sign(x) * y

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        a, r = self._eb_abs, self._eb_rel
        x0 = a / r
        y0 = x0 / a
        ay = np.abs(y)
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.where(ay <= y0, ay * a, x0 * np.exp((np.maximum(ay, y0) - y0) * r))
        return np.sign(y) * x

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Floating-point data to be encoded. May be any object supporting
            the new-style buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        return self._encode(buf, None)

    def encode_masked(self, buf: Buffer, mask: Mask) -> bytes:
        """
        Encode the data in `buf`, ignoring the values where `mask` is
        [`True`][True].

        Parameters
        ----------
        buf : Buffer
            Floating-point data to be encoded. May be any object supporting
            the new-style buffer protocol. The values at masked positions are
            unspecified.
        mask : np.ndarray[tuple[int, ...], np.dtype[np.bool]]
            The [boolean][numpy.bool] mask, of the same shape as the data, of
            the values that do not need to be preserved.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        return self._encode(buf, mask)

    def _encode(self, buf: Buffer, mask: None | Mask) -> bytes:
        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        if not np.issubdtype(dtype, np.floating):
            raise TypeError("can only encode floating point values")

        x = a.astype(np.float64)
        check = np.isfinite(x)
        if mask is not None:
            check &= ~np.asarray(mask, dtype=np.bool).reshape(shape)
        x_checked = x[check]
        bound = np.maximum(self._eb_abs, self._eb_rel * np.abs(x_checked))

        y = self._forward(x)

        delta = self._inner_bound * (1.0 - 1e-9)
        for _ in range(16):
            inner = self._inner(delta)
            if mask is not None and isinstance(inner, MaskAwareCodecMixin):
                encoded_buf = inner.encode_masked(y, mask)  # type: ignore
            else:
                encoded_buf = inner.encode(y)
            encoded = numcodecs.compat.ensure_ndarray(encoded_buf)

            y_dec = np.empty(shape, dtype=np.float64)
            if mask is not None and isinstance(inner, MaskAwareCodecMixin):
                y_dec_buf = inner.decode_masked(encoded, mask, out=y_dec)  # type: ignore
            else:
                y_dec_buf = inner.decode(encoded, out=y_dec)
            y_dec = np.asarray(numcodecs.compat.ensure_ndarray(y_dec_buf)).reshape(
                shape
            )

            x_dec = self._inverse(y_dec).astype(dtype).astype(np.float64)
            error = np.abs(x_dec[check] - x_checked)
            excess = float((error / bound).max()) if error.size > 0 else 0.0
            if excess <= 1.0:
                break
            delta *= (1.0 / excess) * (1.0 - 1e-9)
        else:
            raise ValueError(
                f"cannot satisfy the error bound max({self._eb_abs}, {self._eb_rel} * |x|) with dtype {dtype}"
            )

        # message: dtype shape delta encoded-dtype encoded-shape [padding] encoded
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        message.append(np.array([delta], dtype="<f8").tobytes())

        message.append(leb128.u.encode(len(encoded.dtype.str)))
        message.append(encoded.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(encoded.ndim))
        for s in encoded.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with encoded itemsize
        message.append(
            b"\0"
            * (
                encoded.dtype.itemsize
                - (sum(len(m) for m in message) % encoded.itemsize)
            )
        )

        # ensure that the encoded values are encoded in little endian binary
        message.append(encoded.astype(encoded.dtype.newbyteorder("<")).tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        return self._decode(buf, None, out)

    def decode_masked(
        self, buf: Buffer, mask: Mask, out: None | Buffer = None
    ) -> Buffer:
        """
        Decode the data in `buf`, which was encoded with the same `mask`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        mask : np.ndarray[tuple[int, ...], np.dtype[np.bool]]
            The [boolean][numpy.bool] mask, of the same shape as the decoded
            data, that was passed to `encode_masked`.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol. The values at masked positions are unspecified.
        """

        return self._decode(buf, mask, out)

    def _decode(self, buf: Buffer, mask: None | Mask, out: None | Buffer) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        # message: dtype shape delta encoded-dtype encoded-shape [padding] encoded
        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )

        (delta,) = np.frombuffer(b_io.read(8), dtype="<f8", count=1)

        encoded_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        encoded_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        encoded_size = reduce(lambda a, b: a * b, encoded_shape, 1)

        # remove padding to align with encoded itemsize
        b_io.read(encoded_dtype.itemsize - (b_io.tell() % encoded_dtype.itemsize))

        encoded = (
            np.frombuffer(
                b_io.read(encoded_size * encoded_dtype.itemsize),
                dtype=encoded_dtype.newbyteorder("<"),
                count=encoded_size,
            )
            .astype(encoded_dtype)
            .reshape(encoded_shape)
        )

        inner = self._inner(float(delta))
        y = np.empty(shape, dtype=np.float64)
        if mask is not None and isinstance(inner, MaskAwareCodecMixin):
            y_buf = inner.decode_masked(encoded, mask, out=y)  # type: ignore
        else:
            y_buf = inner.decode(encoded, out=y)
        y = np.asarray(numcodecs.compat.ensure_ndarray(y_buf)).reshape(shape)

        decoded = self._inverse(y).astype(dtype)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this meta-codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this meta-codec.
        """

        return dict(
            id=type(self).codec_id,
            eb_abs=self._eb_abs,
            eb_rel=self._eb_rel,
            codec=copy.deepcopy(self._codec),
            eb_abs_marker=self._eb_abs_marker,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eb_abs={self._eb_abs!r}, eb_rel={self._eb_rel!r}, codec={self._codec!r}, eb_abs_marker={self._eb_abs_marker!r})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "AbsOrRelErrorBoundedCodec":
        """
        Apply the `mapper` to the inner codec (instantiated with a placeholder
        error bound, mapped, and converted back to a configuration in which
        the bound is replaced by the `eb_abs_marker` again).

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that is applied to the inner codec.

        Returns
        -------
        mapped : AbsOrRelErrorBoundedCodec
            The mapped meta-codec.
        """

        placeholder = 0x7EB_A85_5EED * 1e-300  # unlikely to occur in a config
        mapped = mapper(self._inner(placeholder)).get_config()
        config = _replace_marker(mapped, placeholder, self._eb_abs_marker)
        assert isinstance(config, dict)
        return AbsOrRelErrorBoundedCodec(
            eb_abs=self._eb_abs,
            eb_rel=self._eb_rel,
            codec=config,
            eb_abs_marker=self._eb_abs_marker,
        )


numcodecs.registry.register_codec(AbsOrRelErrorBoundedCodec)
