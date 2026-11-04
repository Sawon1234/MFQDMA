"""Minimal, dependency-free reader for R .rda/.rds files (XDR serialization,
versions 2 and 3) sufficient for numeric xts/zoo objects, data frames and
plain vectors.  Pure data parsing: nothing in the file is ever executed.

Usage:  objs = read_rda(path)  ->  {name: RObj}
        to_series(obj)         ->  (dates[np.datetime64], values[2D], colnames)
"""
import bz2, gzip, lzma, struct
import numpy as np


class RObj:
    __slots__ = ('type', 'value', 'attr')

    def __init__(self, t, v, a=None):
        self.type, self.value, self.attr = t, v, a or {}

    def __repr__(self):
        return f'RObj(type={self.type}, attr={list(self.attr)})'


def _decompress(raw):
    if raw[:2] == b'\x1f\x8b':
        return gzip.decompress(raw)
    if raw[:3] == b'BZh':
        return bz2.decompress(raw)
    if raw[:6] == b'\xfd7zXZ\x00':
        return lzma.decompress(raw)
    return raw


class _Reader:
    def __init__(self, buf):
        self.b, self.p, self.refs = buf, 0, []

    def int(self):
        v = struct.unpack_from('>i', self.b, self.p)[0]; self.p += 4; return v

    def length(self):
        n = self.int()
        if n == -1:
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def bytes(self, n):
        v = self.b[self.p:self.p + n]; self.p += n; return v

    # ------------------------------------------------------------------
    def attrs_from_pairlist(self, pl):
        out = {}
        while isinstance(pl, RObj) and pl.type == 2:
            car, tag, cdr = pl.value
            out[tag] = car
            pl = cdr
        return out

    def item(self):
        flags = self.int()
        t = flags & 0xFF
        has_attr, has_tag = bool(flags & (1 << 9)), bool(flags & (1 << 10))
        if t == 254:                       # NILVALUE
            return None
        if t in (253, 242, 241, 252, 251, 250):   # special environments / markers
            return RObj(t, None)
        if t == 255:                       # REFSXP
            idx = flags >> 8
            if idx == 0:
                idx = self.int()
            return self.refs[idx - 1]
        if t in (248, 249, 247):           # package / namespace / persistent
            self.int()                     # 0
            n = self.int()
            names = [self.item() for _ in range(n)]
            o = RObj(t, names); self.refs.append(o); return o
        if t == 1:                         # SYMSXP
            s = self.item()
            o = s.value if isinstance(s, RObj) else s
            self.refs.append(o)
            return o
        if t == 4:                         # ENVSXP
            self.int()
            o = RObj(4, None); self.refs.append(o)
            enclos, frame, hashtab, attrib = (self.item() for _ in range(4))
            o.value = (frame, hashtab)
            return o
        if t in (2, 3, 5, 6, 17, 239, 240):   # pairlist-like
            attr = self.attrs_from_pairlist(self.item()) if has_attr else {}
            tag = self.item() if has_tag else None
            car = self.item()
            cdr = self.item()
            return RObj(2 if t in (239,) else t if t != 240 else 6,
                        (car, tag, cdr), attr)
        if t == 9:                         # CHARSXP
            n = self.int()
            return None if n == -1 else self.bytes(n).decode('utf-8', 'replace')
        if t in (10, 13):                  # LGLSXP, INTSXP
            n = self.length()
            v = np.frombuffer(self.bytes(4 * n), '>i4').astype(np.int64)
            v = np.where(v == -2147483648, np.iinfo(np.int64).min, v)
        elif t == 14:                      # REALSXP
            n = self.length()
            v = np.frombuffer(self.bytes(8 * n), '>f8').astype(float)
        elif t == 15:
            n = self.length()
            v = np.frombuffer(self.bytes(16 * n), '>f8').astype(float).view(complex)
        elif t == 16:                      # STRSXP
            n = self.length()
            v = [self.item() for _ in range(n)]
        elif t in (19, 20):                # VECSXP, EXPRSXP
            n = self.length()
            v = [self.item() for _ in range(n)]
        elif t == 24:                      # RAWSXP
            n = self.length()
            v = self.bytes(n)
        elif t == 25:                      # S4SXP
            v = None
        elif t == 238:                     # ALTREP
            info = self.item(); state = self.item(); attr = self.item()
            cls = info.value[0] if isinstance(info, RObj) else None
            v = self._altrep(cls, state)
            o = RObj(14, v, self.attrs_from_pairlist(attr))
            return o
        else:
            raise ValueError(f'unsupported SEXP type {t} at byte {self.p}')
        attr = self.attrs_from_pairlist(self.item()) if has_attr else {}
        return RObj(t, v, attr)

    @staticmethod
    def _altrep(cls, state):
        if cls in ('compact_intseq', 'compact_realseq'):
            n, start, inc = (state.value if isinstance(state, RObj) else state)[:3]
            return start + inc * np.arange(int(n))
        if cls and cls.startswith('wrap_'):
            return state.value[0].value
        if cls == 'deferred_string':
            return state.value[0].value
        raise ValueError(f'unsupported ALTREP class {cls}')


def _parse(buf):
    if buf[:5] in (b'RDX2\n', b'RDX3\n'):
        buf = buf[5:]
        rda = True
    else:
        rda = False
    assert buf[:2] == b'X\n', 'only XDR format supported'
    r = _Reader(buf); r.p = 2
    version = r.int(); r.int(); r.int()
    if version == 3:
        r.bytes(r.int())
    top = r.item()
    if not rda:
        return {'object': top}
    out = {}
    while isinstance(top, RObj) and top.type == 2:
        car, tag, cdr = top.value
        out[tag] = car
        top = cdr
    return out


def read_rda(path):
    return _parse(_decompress(open(path, 'rb').read()))


def to_series(obj):
    """xts/zoo -> (dates, values (T x k), column names)."""
    vals = np.asarray(obj.value, float)
    dim = obj.attr.get('dim')
    if dim is not None:
        nr, nc = [int(d) for d in dim.value]
        vals = vals.reshape(nc, nr).T          # R is column-major
    else:
        vals = vals[:, None]
    cols = None
    dn = obj.attr.get('dimnames')
    if dn is not None and dn.value and dn.value[1] is not None:
        cols = list(dn.value[1].value)
    idx = obj.attr.get('index')
    if idx is None:
        raise ValueError('no time index (not an xts object?)')
    iv = np.asarray(idx.value, float)
    tclass = idx.attr.get('tclass') or obj.attr.get('.indexCLASS')
    tcl = list(tclass.value) if isinstance(tclass, RObj) else []
    if 'Date' in tcl and iv.max() < 1e6:
        dates = np.datetime64('1970-01-01') + iv.astype('int64').astype('timedelta64[D]')
    else:                                      # POSIXct seconds
        dates = (np.datetime64('1970-01-01T00:00:00') +
                 iv.astype('int64').astype('timedelta64[s]')).astype('datetime64[D]')
    return dates, vals, cols
