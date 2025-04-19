from __future__ import annotations
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
  from ordinal import Ord, FdmtSeq

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
  from ordinal import Ord
  ords = []
  for part in partition_v_list(s):
    if part.startswith('v(') and part.endswith(')'):
      ords.append(parse_v_list(part, **kwargs))
    else:
      ords.append(Ord.from_str(part, **kwargs))

  is_pos = (isinstance(o, Ord) and o.is_pos() for o in ords)
  if any(is_pos):
    assert all(is_pos), f'{s}\n{ords}'
    return VeblenTF.from_ord_list(*cast(List, ord), **kwargs)
  return Veblen(*ords, **kwargs)

class VeblenBase:
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

  def __eq__(self, other):
    return isinstance(other, Veblen) and \
           all(v == o for v, o in zip(self.param, other.param))

  # !! copy
  def idxs_missing(self):
    return [i for i, val in enumerate(self.param) if val is None]

  def is_missing_one(self):
    return len(self.idxs_missing()) == 1

  def make_combined(self, other) -> Ord:
    idxs = self.idxs_missing()
    assert len(idxs) == 1, self
    idx = idxs[0]
    new_params = list(self.param)
    new_params[idx] = other
    ret = Veblen(*new_params)
    from ordinal import Ord
    return Ord(ret)

  def to_latex(self):
    if self.is_binary() and not self.latex_force_veblen:
      a = self.param[0].token.v
      if isinstance(a, int) and 0 <= a <= 3:
        ax = [r'\omega', r'\varepsilon', r'\xi', r'\eta'][a]
        gx = self.param[1].to_latex()
        if a == 0:
          return f'{ax}^{{{gx}}}'  # latex_add_parentheses(gx)
        return f'{ax}_{{{gx}}}'
    elif self.arity() == 3 and self.param[0] == 1 and self.param[1] == 0:
      # v(1,0,a) = G_{a}
      return f'\\Gamma_{{{self.param[-1].to_latex()}}}'

    return '\\varphi({})'.format(', '.join(o.to_latex() for o in self.param))

  def arity(self):
    return len(self.param)

  def is_binary(self):
    return self.arity() == 2

  def recurse_at(self, idx, func):
    sub_node = self.param[idx]
    remain   = copy(self)
    self.param[idx] = None
    return func(sub_node, remain)

  # to (S, ax, Z, gx)
  def partition(self) -> Tuple[Tuple[Ord,...], Ord | None, List[Ord], Ord]:

    assert self.arity() > 0
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

    gx = self.param[-1]     # last term, g or g+1
    if self.is_binary() and \
       gx == 0 and n == 0:  # R4: v(a, 0)[0] = 0
      recorder.skip_next()
      return succ(Ord(0))

    def succ_v(v: Tuple | Ord, nxt, *, n_nxt=n):
      return (True,
              (Ord(Veblen(*v)) if isinstance(v, tuple) else v),
              FdmtSeq(nxt, n_nxt))

    S, ax, Z, gx = self.partition()

    if ax is None:  # R1 R2
      if self.is_binary():
        recorder.skip_next()  # already shown like this
      return succ(Ord('^', 'w', gx))
    if ax.is_succ_ordinal():  # R3-5
      if gx == 0:  # R3: v(S,a+1,Z,0) : b -> v(S,a,b,Z)
        # v(S,a+1,Z,0)[0] = 0
        if n == 0:
          recorder.skip_next()
          return succ(Ord(0))
        # v(S,a+1,Z,0)[n+1] = v(S,a,v(S,a+1,Z,0)[n],Z)
        return succ_v((*S, ax.dec(), None, *Z),
                                     self,
                      n_nxt=n-1)
      if gx.is_succ_ordinal():  # R4: v(S,a+1,Z,g+1): b -> v(S,a,b,Z)
        # R4-1 (binary R6) v(S,a+1,Z,g+1)[0]   = v(S,a+1,Z,g) + 1
        if n == 0:
          recorder.skip_next()
          return succ(Veblen(*S, ax, *Z, gx.dec()) + 1)
        # R4-2 (binary R7) v(S,a+1,Z,g+1)[n+1] = v(S,a,v(S,a+1,Z,g+1)[n],Z)
        return succ_v((*S, ax.dec(), None, *Z),
                                     self,
                      n_nxt=n-1)
      else:  # R5 (binary R3) v(S,a+1,Z,g[n])
        return succ_v((*S, ax, *Z, None),
                                   gx)

    # R6-8: ax is LO
    if gx == 0:  # R6 v(S,a,Z,0)[n] = v(S,a[n],Z,0)
      return succ_v((*S, None, *Z, 0),
                         ax)
    # R7 (binary R9) v(S,a,Z,g+1)[n] = v(S,a[n],Z,(S,a,Z,g)+1)
    elif gx.is_succ_ordinal():
      recorder.skip_next()
      return succ_v((*S, None, *Z, Veblen(*S, ax, *Z, gx.dec()) + 1),
                         ax)
    # R8 (binary R5) v(S,a,Z,g[n])
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

# val@pos
# todo: just inherit from Ord
class OrdPos:
  val: Ord
  pos: Ord

  def __init__(self, val, pos):
    from ordinal import Ord
    self.val = Ord.from_any(val)
    self.pos = Ord.from_any(pos)

  def __eq__(self, other):
    return self.val == other.val and \
           self.pos == other.pos

class VeblenTF(VeblenBase):
  param: Tuple[OrdPos, ...]  # v:       v(1@w, 1@0)
                             # index:       0    1

  def __init__(self, *args : OrdPos, **kwargs):
    assert len(args) > 0
    while len(args) > 2 and args[0].val == 0:  # drop 0 at beginning
      args = args[1:]
    super().__init__(*args, **kwargs)

  @classmethod
  def from_ord_list(cls, *args : Ord, **kwargs) -> Self:
    return cls(*(o.to_pos() for o in args), **kwargs)

  def __eq__(self, other):
    # todo 2: eq V and V-TF
    return isinstance(other, Self) and \
           all(v == o for v, o in zip(self.param, other.param))
