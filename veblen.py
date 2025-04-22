from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from enum import Enum
from html_utils import OutType
from typing import Any, Dict, Generator, List, Self, Tuple, cast, TYPE_CHECKING
import re
import utils
from utils import Recorder

from html_utils import contact_request
import ordinal

if TYPE_CHECKING:
  from ordinal import FdmtSeq, Ord, OrdPos

def strip_pre_post(pre: str, s: str, post: str = '') -> str:
  l1, l2 = len(pre), len(post)
  assert len(s) >= l1 + l2, f'{s} {pre} {post}'
  assert s.startswith(pre) and s.endswith(post), f'{pre} {s} {post}'
  return s[l1:-l2] if l2 > 0 else s[l1:]

def partition_ord_list(s_: str) -> Generator[str, None, None]:
  s = strip_pre_post('(', s_.replace(' ', ''), ')')
  depth = 0
  last = 0
  for i, c in enumerate(s):
    if c == '(':
      depth += 1
    elif c == ')':
      depth -= 1
    elif c == ',' and depth == 0:
      yield s[last:i]
      last = i + 1
  yield s[last:]

def partition_v_list(s_: str) -> Generator[str, None, None]:
  s = strip_pre_post('v', s_)
  for t in partition_ord_list(s):
    yield t

def parse_v_list(s: str, **kwargs) -> Veblen | VeblenTF:
  ords = []
  for part in partition_v_list(s):
    if part.startswith('v(') and part.endswith(')'):
      ords.append(parse_v_list(part, **kwargs))
    else:
      ords.append(ordinal.Ord.from_str(part, **kwargs))

  is_pos = (isinstance(o, ordinal.Ord) and o.is_pos() for o in ords)
  if any(is_pos):
    assert all(is_pos), f'{s}\n{ords}'
    return VeblenTF.from_ord_list(*cast(List, ords), **kwargs)
  return Veblen(*ords, **kwargs)

class VeblenBase(ABC):
  param: Tuple
  latex_force_veblen: bool  # force showing forms like v(0, 1) in latex

  def __init__(self, *args, latex_force_veblen=False):
    self.param = args
    self.latex_force_veblen = latex_force_veblen

  def __add__(self, rhs):
    from ordinal import Ord
    return Ord('+', self, rhs)

  def __str__(self):
    return 'v({})'.format(','.join(
      (str(o) if o is not None else '.' for o in self.param)
    ))

  def __repr__(self):
    return self.__str__()

  def to_latex(self) -> str:
    return '\\varphi({})'.format(', '.join(o.to_latex() for o in self.param))

  @abstractmethod
  def idxs_missing(self) -> List[int]:
    pass

  def is_missing_one(self) -> bool:
    return len(self.idxs_missing()) == 1

  def get_only_idx_missing(self) -> int:
    idxs = self.idxs_missing()
    assert len(idxs) == 1, self
    return idxs[0]

  @abstractmethod
  def make_combined(self, other) -> Ord:
    pass

  def recurse_at(self, idx, func):
    sub_node = self.param[idx]
    remain   = copy(self)
    self.param[idx] = None
    return func(sub_node, remain)

class Veblen(VeblenBase):
  param: Tuple[Ord, ...]  # v:       v(1, 0, 0)
                          # index:     0  1  2

  def __init__(self, *args, **kwargs):
    assert len(args) > 0
    while len(args) > 2 and args[0] == 0:  # drop 0 at beginning
      args = args[1:]
    from ordinal import Ord
    super().__init__(*(Ord.from_any(o) for o in args), **kwargs)

  # not used
  @classmethod
  def from_nested(cls, *args, latex_force_veblen=False) -> Veblen:
    param = []
    for a in args:
      match a:
        case tuple():
          param.append(Ord(cls.from_nested(*a, latex_force_veblen=latex_force_veblen)))
        case _:
          param.append(Ord.from_any(a, latex_force_veblen=latex_force_veblen))
    return Veblen(*param, latex_force_veblen=latex_force_veblen)

  def rm_zero(self):
    self.param = Veblen.zero_removed(*self.param)

  @staticmethod
  def zero_removed(*args: Ord):
    while len(args) > 1 and args[0] == 0:
      args = args[1:]
    return tuple(args)

  def __eq__(self, other):
    self.rm_zero()
    match other:
      case Veblen():
        other.rm_zero()
        if self.math_arity() != other.math_arity():
          return False
        return all(v == o for v, o in zip(self.param ,
                                          other.param))
      case VeblenTF():
        return other.__eq__(self)
      case ordinal.FdmtSeq():  # structual, no eval
        return False
      case int():
        return False  # # structual, no eval
        if other != 1:  # only w^0 == 1 could equal
          return False
        return all(v == 0 for v in self.param)
      case _:
        return NotImplemented

  def idxs_missing(self) -> List:
    return [i for i, ord in enumerate(self.param) if ord is None]

  def make_combined(self, other) -> Ord:
    new_params = list(self.param)
    new_params[self.get_only_idx_missing()] = other
    return ordinal.Ord(Veblen(*new_params))

  def to_latex(self):
    arity = self.math_arity()
    if arity == 1 and not self.latex_force_veblen:
      return f'\\omega^{{{self.param[0].to_latex()}}}'
    if arity == 2 and not self.latex_force_veblen:
      a = self.param[0].token.v
      if isinstance(a, int) and 0 <= a <= 3:
        ax = [r'\omega', r'\varepsilon', r'\xi', r'\eta'][a]
        gx = self.param[1].to_latex()
        if a == 0:
          return f'{ax}^{{{gx}}}'  # latex_add_parentheses(gx)
        return f'{ax}_{{{gx}}}'
    elif arity == 3 and self.param[0] == 1 and self.param[1] == 0:
      # v(1,0,a) = G_{a}
      return f'\\Gamma_{{{self.param[-1].to_latex()}}}'

    # default case
    return super().to_latex()

  # emphasize it's math, not coding (diff for V-TF)
  def math_arity(self):
    n = len(self.param)
    assert n > 0, self
    return n

  def is_math_binary(self):
    return self.math_arity() == 2

  # to (S, ax, Z, gx)
  def partition(self) -> Tuple[Tuple[Ord,...], Ord | None, List[Ord], Ord]:

    assert len(self.param) > 0
    gx = self.param[-1]
    remain = self.param[:-1]

    Z : List[Ord] = []
    while len(remain) > 0 and remain[-1] == 0:
      Z.append(remain[-1])
      remain = remain[:-1]

    if len(remain) > 0:
      return remain[:-1], remain[-1], Z, gx
    return (), None, Z, gx

  def index(self, n : int, recorder: Recorder) -> Tuple[bool, Ord | None, FdmtSeq]:
    from ordinal import Ord, FdmtSeq

    def succ(nxt, remain=None):
      return (True, remain, FdmtSeq(nxt, n))

    def succ_v(v: Tuple | Ord, nxt, *, n_nxt=n):
      return (True,
              (Ord(Veblen(*v)) if isinstance(v, tuple) else v),
              FdmtSeq(nxt, n_nxt))

    gx = self.param[-1]     # last term, g or g+1

    # R4: v(a, 0)[0] = 0
    if self.is_math_binary() and \
       gx == 0 and n == 0:
      recorder.skip_next()
      return succ(Ord(0))

    S, ax, Z, gx = self.partition()

    if ax is None:  # R1 R2
      if self.is_math_binary():  # (0,g)
        recorder.skip_next()  # already shown like this
      return succ(Ord('^', 'w', gx))
    if ax.is_succ_ordinal():  # R3-5
      if gx == 0:  # R3: v(S,a+1,Z,0) : x -> v(S,a,x,Z)
        # v(S,a+1,Z,0)[0] = 0
        if n == 0:
          recorder.skip_next()
          return succ(Ord(0))
        # v(S,a+1,Z,0)[n+1] = v(S,a,v(S,a+1,Z,0)[n],Z)
        return succ_v((*S, ax.dec(), None, *Z),
                                     self,
                      n_nxt=n-1)
      if gx.is_succ_ordinal():  # R4: v(S,a+1,Z,g+1): x -> v(S,a,x,Z)
        # R4-1 (binary R6) v(S,a+1,Z,g+1)[0]   = v(S,a+1,Z,g) + 1
        if n == 0:
          recorder.skip_next()
          return succ(Veblen(*S, ax, *Z, gx.dec()) + 1)
        # R4-2 (binary R7) v(S,a+1,Z,g+1)[n+1] = v(S,a,v(S,a+1,Z,g+1)[n],Z)
        return succ_v((*S, ax.dec(), None, *Z),
                                     self,
                      n_nxt=n-1)
      # R5=R8 (binary R3) v(S,a+1,Z,g[n])
      else:
        return succ_v((*S, ax, *Z, None),
                                   gx)

    # R6-8: ax is LO
    if gx == 0:  # R6 v(S,a,Z,0)[n] = v(S,a[n],Z,0)
      return succ_v((*S, None, *Z, 0),
                         ax)
    # R7 (binary R9) v(S,a,Z,g+1)[n] = v(S,a[n],(S,a,g,Z)+1,Z)
    # wiki 2.7, book R7 is WRONG and not match V(@)!
    elif gx.is_succ_ordinal():
      recorder.skip_next()
      return succ_v((*S, None, Veblen(*S, ax, *Z, gx.dec()) + 1, *Z),
                         ax)
    # R8=R5 (binary R5) v(S,a,Z,g[n])
    return succ_v((*S, ax, *Z, None),
                               gx)

    # old binary calc
    ax = self.param[0]   # first non-zero term except last term. a or a+1

    if ax == 0:  # R2: v(0,g) = w^g (for any g)
      # rec.skip_next()
      return succ(Ord('^', 'w', gx))
    if gx.is_limit_ordinal():  # R3, g is LO
      return succ_v((ax, None), gx)
    if ax.is_limit_ordinal():  # R8-9 v(a, .)
      if gx == 0:  # R8 v(a, 0)
        return succ_v((None, 0), ax)
      else:  # R9 v(a, g+1)
        return succ_v((None, Veblen(ax, gx.dec()) + 1), ax)
    else:  # R5-7 v(a+1, .)
      a = ax.dec()
      if gx == 0:  # R5 v(a+1,0) : g -> v(a, g)
        return succ_v((a, None), Veblen(ax, 0), n_nxt=n-1)
      else:
        if n == 0:  # R6 v(a+1,g+1)[0] = v(a+1,g)+1
          # e.g. e1[0] = e0 + 1
          recorder.skip_next()
          return succ(Veblen(ax, gx.dec()) + 1)
        else:  # R7 v(a+1,g+1)[n+1]: g -> v(a, g)
          # e.g. e2 = {e1+1, w^(...), w^w^(...), }
          return succ_v((a, None), self, n_nxt=n-1)

# TransFinitary Veblen
# note: order must be from larger Ord to smaller
class VeblenTF(VeblenBase):
  param: Tuple[OrdPos, ...]  # v:       v(1@w, 1@0)
                             # index:       0    1

  # val only allow 0, when it's 0@0
  def __init__(self, *args_ : OrdPos, **kwargs):
    args = VeblenTF.zero_removed(*args_)
    if len(args) == 0:
      args = [ordinal.OrdPos(0,0)]
    super().__init__(*args, **kwargs)

  def rm_zero(self):
    self.param = VeblenTF.zero_removed(*self.param)

  @staticmethod
  def zero_removed(*args: OrdPos):
    result = []
    for o in args:
      assert o.is_pos(), f"Not OrdPos: {o}"
      if o.val() != 0:
        result.append(o)
    return tuple(result)

  @classmethod
  def from_ord_list(cls, *args : Ord, **kwargs) -> Self:
    return cls(*(o.to_pos() for o in args), **kwargs)

  def toVeblen(self) -> Veblen:
    assert self.math_arity().is_natural()
    ans = [ordinal.Ord(0) for _ in range(cast(int, self.math_arity().token.v))]
    for val, pos in self.param:
      i = pos.token.v
      assert isinstance(i, int) and i >= 0
      ans[i] = val
    return Veblen(*ans[::-1])

  def __eq__(self, other):
    self.rm_zero()
    match other:
      case VeblenTF():
        other.rm_zero()
        if len(self.param) != len(other.param):
          return False
        return all(v == o for v, o in zip(self.param,
                                          other.param))
      case Veblen():
        if not self.math_arity().is_natural() or \
           not self.math_arity() == other.math_arity():
          return False
        # todo 3: sml perf: cmp w/o fully expand
        # False if any pos unresolved
        for pos in self.poses():
          if isinstance(pos.token.v, ordinal.FdmtSeq):
            return False
        return self.toVeblen() == other

      case int() | ordinal.FdmtSeq():
        return False

    return NotImplemented

  def to_latex(self):
    arity = self.math_arity()
    if arity.is_natural() and arity.token.v <= 3 and \
       not self.latex_force_veblen:
      return self.toVeblen().to_latex()
    if len(self.param) == 1:
      return super().to_latex()  # shown as v(a@b)
    #  $ \begin{pmatrix} a & b \\ c & d \end{pmatrix} $
    l1 = ' & '.join(o.to_latex() for o in self.vals())
    l2 = ' & '.join(o.to_latex() for o in self.poses())
    return (r'{\begin{pmatrix} ' +
            l1 +
            r' \\ ' +
            l2 +
            r' \end{pmatrix}}'
            )

  def vals(self) -> Generator[Ord, None, None]:
    for ord_pos in self.param:
      yield ord_pos.val()

  def poses(self) -> Generator[Ord, None, None]:
    for ord_pos in self.param:
      yield ord_pos.pos()

  # * note: code idx, not ord idx
  def idxs_missing(self) -> List:
    ret = []
    for i, ord in enumerate(self.param):
      miss = (ord.val() is None) + (ord.pos() is None)
      assert miss != 2
      if miss == 1:
        ret.append(i)
    return ret

  def make_combined(self, other) -> Ord:
    new_params = list(self.param)
    idx = self.get_only_idx_missing()
    new_ord_pos = new_params[idx].make_combined(other, child_only=True)
    assert new_ord_pos is not None
    new_params[idx] = new_ord_pos.to_pos()
    return ordinal.Ord(VeblenTF(*new_params))

  # emphasize it's math, not coding (diff for V-TF)
  def math_arity(self) -> Ord:
    assert len(self.param) > 0
    first = self.param[0]
    if first.val() == 0:
      assert first.pos() == 0
      return ordinal.Ord(1)
    return first.pos() + 1

  def is_math_binary(self):
    return self.math_arity() == 2

  # get (S, a, b, g) from (S, a@b, g@0)
  # must rm 0@x before calling
  def partition(self) -> \
    Tuple[Tuple[OrdPos,...], Ord | None, Ord | None, Ord]:

    def break_last(li: Tuple[OrdPos,...]) \
      -> Tuple[Tuple[OrdPos,...], Ord | None, Ord | None]:
      return li[:-1], li[-1].val(), li[-1].pos()

    assert len(self.param) > 0
    last = self.param[-1]
    if last.pos() == 0:
      if len(self.param) >= 2:
        return *break_last(self.param[:-1]), last.val()
      return (), None, None, last.val()
    return *break_last(self.param), ordinal.Ord(0)

  def index(self, n : int, recorder: Recorder) -> Tuple[bool, Ord | None, FdmtSeq]:
    from ordinal import FdmtSeq, Ord, OrdPos

    def succ(nxt, remain=None):
      return (True, remain, FdmtSeq(nxt, n))

    def succ_v(li: Tuple | Ord, val, *, n_nxt=n):
      return (True,
              (Ord(VeblenTF(*li)) if isinstance(li, tuple) else li),
              FdmtSeq(val, n_nxt))

    S, ax, bx, gx = self.partition()

    def ab():
      return ax @ bx

    # R8 - resolve
    # if isinstance(bx, FdmtSeq):
    #   return succ_v((*S, OrdPos(ax, None), OrdPos(gx, 0)),
    #                                 bx,
    #                  n_nxt=bx.n)

    # binary R4: v(a, 0)[0] = 0
    if self.is_math_binary() and \
       gx == 0 and n == 0:
      recorder.skip_next()
      return succ(Ord(0))

    # R1: v(g) = w^g
    if ax is None:
      assert len(S) == 0
      if self.is_math_binary():
        recorder.skip_next()  # already shown like this
      return succ(Ord('^', 'w', gx))

    # R2: v(...,g[n])
    # wiki 3.3; MV R5,R8
    if gx.is_limit_ordinal():
      return succ_v((*S, ab(), OrdPos(None, 0)),
                                      gx)

    # R3-6
    if ax.is_succ_ordinal():
      assert bx is not None
      # R3-4
      if bx.is_succ_ordinal():
        # R3: v(a+1@b+1) : x -> v(a@b+1, x@b)
        # wiki 3.1; MV R3
        if gx == 0:
          if n == 0:
            recorder.skip_next()
            # wiki 3.1.1
            return succ(Ord(0))
          # v(a+1@b+1)[n+1] = v(a@b+1, v(S,a+1@b+1)[n]@b)
          # wiki 3.1.2
          return succ_v((*S, ax.dec() @ bx, OrdPos(None, bx.dec())),
                                                   self,
                        n_nxt=n-1)

        # R4: v(a+1@b+1, g+1@0): x -> v(a@b+1, x@b)
        # wiki 3.2; MV R4
        assert gx.is_succ_ordinal()
        if n == 0:
          recorder.skip_next()
          # v(a+1@b+1,g+1@0)[0] = v(a+1@b+1,g@0)+1
          # wiki 3.2.1
          return succ(VeblenTF(*S, ab(), gx.dec()@0) + 1)

        # v(S, a+1@b+1, g+1)[n+1] = v(S, a@b+1, v(S, a+1@b+1,g+1)[n] @ b)
        # wiki 3.2.2
        return succ_v((*S, ax.dec() @ bx, OrdPos(None, bx.dec())),
                                                 self,
                       n_nxt=n-1)

      # R5-6, b is LO
      # R5 v(S, a+1@b)[n] = v(S, a@b, 1@b[n])
      # wiki 3.6
      if gx == 0:
        return succ_v((*S, ax.dec() @ bx, OrdPos(1, None)),
                                                    bx)
      # R6 v(a+1@b, g+1) = v(a@b, v(a+1@b, g)+1 @ b[n])
      # wiki 3.7
      return succ_v((*S,
                     ax.dec() @ bx,
                     OrdPos(VeblenTF(*S, ab(), gx.dec() @ 0) + 1,
                            None)),
                            bx)

    # R7-8 ax is LO
    # R7 v(a[n]@b)
    # (wiki 3.4, 3.8; MV R6)
    if gx == 0:
      return succ_v((*S, OrdPos(None, bx)),
                                ax)

    # Rx a@b+1, g+1@0
    # (wiki 3.5)
    assert bx is not None
    if bx.is_succ_ordinal():
      return succ_v((*S, OrdPos(None, bx),
                    (VeblenTF(*S, ab(), gx.dec() @ 0) + 1) @ bx.dec()),
                                ax)

    # R8 v(a@b, g+1)[n] = v(a[n]@b, v(a@b,g)+1 @ b[n])
    # (wiki 3.9; MV R7)
    assert gx.is_succ_ordinal()
    return succ_v((*S, OrdPos(FdmtSeq(ax, n), bx),
                   OrdPos(VeblenTF(*S, ab(), gx.dec() @ 0) + 1,
                          None)),
                          bx)
