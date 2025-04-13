from __future__ import annotations
from copy import deepcopy
from enum import Enum
from typing import List, Dict, Tuple, cast
import re

from html_utils import contact_request

"""
todo:
"""

# * globals
debug_mode = False
rotate_counter : int = 0

def ord_set_debug_mode(new : bool):
  global debug_mode
  debug_mode = new

def get_rotate_counter() -> int:
  global rotate_counter
  return rotate_counter

# * Exceptions
class KnownError(Exception):
  @staticmethod
  def raise_if(cond, msg):
    if cond:
      raise KnownError(msg)

def kraise(*args):
  return KnownError.raise_if(*args)

def strip_pre_post(pre: str, s: str, post: str) -> str:
  l1, l2 = len(pre), len(post)
  assert len(s) >= l1 + l2, f'{s} {pre} {post}'
  assert s.startswith(pre) and s.endswith(post), f'{s} {pre} {post}'
  return s[l1:-l2]

class Veblen:
  param: List[Ord]  # v:       v(1, 0, 0)
                    # index:   0  1  2

  def __init__(self, *args):
    assert len(args) >= 2
    self.param = [Ord.from_any(o) for o in args]

  @staticmethod
  def from_str(s_ : str) -> Veblen:
    s = strip_pre_post('v(', s_, ')').strip()
    ords = map(Ord.from_str, s.split(','))
    return Veblen(*ords)

  def __eq__(self, other):
    # todo: consider complex Veblen
    return isinstance(other, Veblen) and \
           all(v == o for v, o in zip(self.param, other.param))

  def __add__(self, rhs):
    return Ord('+', self, rhs)

  def __str__(self):
    return 'v({})'.format(', '.join(map(str, self.param)))

  def to_latex(self):
    return '\\varphi({})'.format(', '.join(o.to_latex() for o in self.param))

  def arity(self):
    return len(self.param)

  def is_binary(self):
    return self.arity() == 2

  def index(self, n : int, rec: Recorder) -> Ord:
    # todo 1: separate cnt for steps discarded before '...'
    if rec.inc_discard_check_limit():
      return Ord(self)

    assert self.is_binary(), f'WIP {self}'
    ax = self.param[0]   # first non-zero term except last term. a or a+1
    gx = self.param[-1]  # last term, g or g+1

    if gx == Ord(0) and n == 0:  # R4: v(a, 0)[0] = 0
      return Ord(0)
    if ax == Ord(0):  # R2: v(0, g) = w^g
      return Ord('^', Ord('w'), gx)
    if gx.is_limit_ordinal():  # R3, g is LO
      # todo 1: record expansion, pass in rec
      return Ord(Veblen(ax, gx.fs_at(n, rec)))
    if ax.is_limit_ordinal():  # R8-9 v(a, .)
      if gx == Ord(0):  # R8 v(a, 0)
        # todo 1: record
        return Ord(Veblen(ax.fs_at(n, rec), 0))
      else:  # R9 v(a, g+1)
        return Ord(Veblen(ax.fs_at(n, rec),
                          Veblen(ax, gx.dec()) + 1))
    else:  # R5-7 v(a+1, .)
      a = ax.dec()
      if gx == 0:  # R5 v(a+1,0) : g -> v(a, g)
        # todo 1: record
        return Ord(Veblen(a, Veblen(ax, Ord(0)).index(n-1, rec)))
      else:
        if n == 0:  # R6 v(a+1,g+1)[0] = v(a+1,g)+1
          # e.g. e1[0] = e0 + 1
          return Veblen(ax, gx.dec()) + 1
        else:  # R7 v(a+1,g+1)[n+1]: g -> v(a, g)
          # e.g. e2 = {e1+1, w^(...), w^w^(...), }
          return Ord(Veblen(a, self.index(n-1, rec)))



# number, w, e, operators, Veblen
class Token:
  v: int | str | Veblen
  ord_maps : Dict[str, str] = {
    'w': r'\omega',
    'e': r'\varepsilon_{0}',
    'x': r'\xi_{0}',
    'h': r'\eta_{0}',
  }
  latex_maps : Dict[str, str] = {
    **ord_maps,
    '+': '+',
    '*': r'\cdot',
    '^': '^',
  }

  def __init__(self, v):
    match v:
      case Token():
        self.v = v.v
      case int() | Veblen():
        self.v = v
      case str():
        assert len(v) > 0
        if v.isdigit():
          self.v = int(v)
        elif v[0] == 'v':
          self.v = Veblen.from_str(v)
        else:
          self.v = v
      case _:
        assert 0, v

  def __eq__(self, other_):
    other = other_ if isinstance(other_, Token) else Token(other_)
    return type(self.v) == type(other.v) and self.v == other.v

  def __str__(self):
    return str(self.v)

  def __repr__(self):
    return self.__str__()

  def is_natural(self):
    return isinstance(self.v, int)

  def get_natrual(self) -> int:
    assert self.is_natural()
    return cast(int, self.v)

  def to_latex(self):
    if self.is_natural():
      return str(self.v)
    if isinstance(self.v, Veblen):
      return self.v.to_latex()
    elif self.v in Token.latex_maps.keys():
      return Token.latex_maps[self.v]
    assert 0, self

class Recorder:
  rec_limit : int
  cal_limit : int
  data      : List  # allow extra elem for result
  full      : bool
  n_discard : int
  n_pre_discard : int  # discarded before reach limit
  will_skip_next : bool  # skip the next record

  def __init__(self, rec_limit, cal_limit):
    assert rec_limit >= 2 or rec_limit == 0  # 0 for inactive
    self.rec_limit = rec_limit
    self.cal_limit = cal_limit
    self.data      = []
    self.full      = False
    self.n_discard = 0
    self.n_pre_discard = 0
    self.will_skip_next = False

  def active(self):
    return self.rec_limit != 0

  def tot_discard(self):
    return self.n_discard + self.n_pre_discard

  def cal_limit_reached(self):
    return len(self.data) + self.tot_discard() >= self.cal_limit

  def no_mid_steps(self):
    return self.rec_limit == 2

  def record(self, entry, res=False):
    try:
      if not self.active():
        return
      if self.full:
        self.n_discard += 1
        if res:  # force replace
          self.data[-1] = entry
      else:
        self.data.append(entry)
        assert len(self.data) <= self.rec_limit
        if len(self.data) == self.rec_limit - 1:  # save 1 for result
          self.full = True
    finally:
      self.will_skip_next = False

  # return True if cal_limit reached
  def inc_discard_check_limit(self) -> bool:
    if self.full:
      self.n_discard += 1
    else:
      self.n_pre_discard += 1
    return self.cal_limit_reached()

  def skip_next(self):
    self.will_skip_next = True

  def to_latex(self, entry_to_latex):
    ret = r' \begin{align*}' + '\n'
    ret += entry_to_latex(self.data[0])
    for ord in self.data[1:-1]:
      ret += f'  &= {entry_to_latex(ord)} \\\\\n'
    if self.n_discard > 1:  # 1 step could be (a+b)[x] = a+b[x]
      ret += r'  &\phantom{=} \vdots \quad \raisebox{0.2em}{\text{' + \
             f'after {self.n_discard} more steps' r'}} \\' + '\n'
    ret += f'  &= {entry_to_latex(self.data[-1])} \\\\\n'
    ret += r'\end{align*} ' + '\n'
    return ret

def test_display(obj, f_calc, f_display, expected=None,
                 limit=100, test_only=False , show_steps=False):
  recorder = Recorder((15 if show_steps else 0), limit)
  res = f_calc(obj, recorder)

  if expected is not None:
    assert res == expected, f'{res} != {expected}'

  if test_only:
    return None

  if not show_steps:
    return f'{f_display(obj)}={f_display(res)}'

  assert recorder is not None
  return recorder.to_latex(f_display)

# Ordinal
ord_rotate_at_init = False
class Ord:
  token: Token  # operator +,*,^ ; natural number str; w,e ; Veblen
  left : Ord
  right: Ord

  def __init__(self, token, left=None, right=None):
    self.token = Token(token)
    self.left  = Ord.from_any(left)
    self.right = Ord.from_any(right)
    if ord_rotate_at_init:
      self.rotate()

  def rotate(self) -> None:
    if self.is_atomic():
      return

    global rotate_counter
    rotate_counter += 1

    while self.token.v == '+' and self.left.token.v == '+':
      #          +                     +
      #        /   \                 /   \
      #      +      r    ===>      ll      +
      #     / \                           / \
      #   ll   lr                       lr    r
      self.left, self.right = \
        Ord.rotated(self.left.left), \
        Ord('+', Ord.rotated(self.left.right), Ord.rotated(self.right))
    else:
      self.left.rotate()
      self.right.rotate()

  @staticmethod
  def rotated(ord : Ord) -> Ord:
    if ord is None:
      return None
    ord.rotate()
    return ord

  def __eq__(self, other):
    if isinstance(other, int):
      return self.token.v == other
    if not ord_rotate_at_init:
      self.rotate()
      other.rotate()
    return self.token == other.token and \
           self.left.__eq__(other.left) and \
           self.right.__eq__(other.right)

  def to_infix(self):
    if self.is_atomic():
      return str(self.token)
    return f"({self.left.to_infix()}{self.token}{self.right.to_infix()})"

  def __str__(self):
    return self.to_infix()

  def __repr__(self):
    return self.__str__()

  def is_atomic(self):
    assert (self.left is None) == (self.right is None), self
    return self.left is None

  def is_valid(self):
    return self.is_atomic() or \
           self.left.is_valid() and self.right.is_valid()

  def is_natural(self):
    return self.token.is_natural()

  @staticmethod
  def from_str(expression: str):
    """
    Read an infix expression and construct the corresponding binary tree.
    """
    kraise(len(expression) == 0, "Can't read Ordinal from empty string")

    def precedence(op):
      if op == "+":
        return 1
      if op == "*":
        return 2
      if op == "^":
        return 3
      return 0

    def is_operand(tok : str):
      return tok.isdigit() or tok in Token.ord_maps or \
             len(tok) > 3 and tok.startswith('v(')

    def to_postfix(tokens):
      """
      Convert the infix expression tokens to postfix (RPN) using Shunting Yard algorithm.
      """
      out = []
      stack = []
      for tok in tokens:
        if is_operand(tok):
          out.append(tok)
        elif tok == "(":
          stack.append(tok)
        elif tok == ")":
          while stack and stack[-1] != "(":
            out.append(stack.pop())
          stack.pop()  # pop the '('
        else:  # Operator
          while (stack and precedence(stack[-1]) >= precedence(tok)):
            out.append(stack.pop())
          stack.append(tok)

      while stack:
        out.append(stack.pop())
      return out

    def build_tree(postfix) -> Ord:
      """
      Build a binary tree from the postfix expression.
      """
      stack = []
      for tok in postfix:
        if is_operand(tok):
          stack.append(Ord(tok))
        else:  # Operator
          right = stack.pop()
          left = stack.pop()
          stack.append(Ord(tok, left, right))

      kraise(len(stack) != 1,
             f"Can't read ordinal from {expression}: " +
             f"len(stack) is {len(stack)}: " + ' '.join(str(ord) for ord in stack)
             if debug_mode
             else f"{len(stack)} terms found. If you believe your input is valid, " +
                  contact_request)
      return stack[0]

    try:
      tokens = re.findall(r'\d+|[+*^()]|v\(.*?\)|' + '|'.join(Token.ord_maps), expression)
      # Convert infix to postfix (RPN)
      postfix = to_postfix(tokens)
      return build_tree(postfix)
    except KnownError as e:
      raise e
    except Exception as e:
      raise KnownError(f"Can't read ordinal from {expression}: " +
                       str(e) if debug_mode else
                       f"If you believe your input is valid, {contact_request}.")

  @staticmethod
  def from_any(expression) -> Ord:
    match expression:
      case None:
        return cast(Ord, None)
      case Ord():
        return expression
      case str():
        return Ord.from_str(expression)
      case int():
        return Ord.from_str(str(expression))
      case Veblen():
        return Ord(expression)
      case _:
        assert 0, expression

  def to_latex(self):
    if self.is_atomic():
      return self.token.to_latex()

    def parentheses_on_demand(node: Ord, out_op: str):
      if node.is_atomic() or \
         (out_op == '*' and node.token.v == '^'):
        return node.to_latex()
      return '{(' + node.to_latex() + ')}'

    match self.token.v:
      case '+':
        return self.left.to_latex()  + \
               self.token.to_latex() + \
               self.right.to_latex()
      case '*':
        return parentheses_on_demand(self.left, '*') + \
               self.token.to_latex() + \
               parentheses_on_demand(self.right, '*')
      case '^':
        return parentheses_on_demand(self.left, '^') + \
               self.token.to_latex() + \
               '{' + self.right.to_latex() + '}'

  def is_limit_ordinal(self):
    if self.is_atomic():
      return not self.is_natural()
    if self.token.v == '+':
      return self.right.is_limit_ordinal()
    # a * b and a ^ b is surely limit ordinal
    assert not self.left.is_natural(), self
    return True

  def dec(self) -> Ord:
    if self.is_atomic():
      i = self.token.get_natrual()
      assert i > 0, self
      return Ord(i - 1)
    else:
      assert self.token.v == '+', self
      if self.right == Ord(1):
        return self.left
      return Ord(self.token, self.left, self.right.dec())

  def simplify(self):
    if self.token.v in '*^' and self.right == Ord(1):
      self.token, self.left, self.right = \
        self.left.token, self.left.left, self.left.right

  @staticmethod
  def simplified(node: Ord):
    node.simplify()
    return node

  fs_cal_limit_default = 500
  def fs_at(self, n, recorder : Recorder) -> Ord:
    return FdmtSeq(self, n).calc(recorder)

# Fundamental Sequence
class FdmtSeq:
  ord: Ord
  n  : int

  def __init__(self, ord, n: int):
    self.ord = Ord.from_any(ord)
    self.n   = n

  def __eq__(self, other):
    return self.ord == other.ord and self.n == other.n

  def __str__(self):
    return f'{self.ord}[{self.n}]'

  def __repr__(self):
    return self.__str__()

  def to_latex(self, always_show_idx=False):
    ret = f'{self.ord.to_latex()}'
    if always_show_idx or self.ord.is_limit_ordinal():
      ret += f'[{self.n}]'
    return ret

  cal_limit_default = 500
  # todo: return FdmtSeq if not end
  def calc(self, recorder : Recorder) -> Ord:

    def impl(ord : Ord, n, rec_pre=None) \
      -> Ord:
      if recorder.active():
        if rec_pre is None:
          recorder.record(ord)
        else:
          assert rec_pre.is_valid()
          recorder.record(Ord('+', rec_pre, ord))

      if recorder.cal_limit_reached():
        return ord

      if ord.is_atomic():
        if ord.is_natural():
          return ord
        match ord.token.v:
          case 'w':
            return Ord(n)
          case 'e':
            return impl(Ord.from_str('w^('*(n-1) + 'w' + ')'*(n-1)), n, rec_pre)
          case Veblen():
            return impl(ord.token.v.index(n, recorder), n, rec_pre)
          case _:
            assert 0, f'{ord} @ {n}'

      # transform w*2 and w^2 so they can be indexed
      def transform(node: Ord) -> Ord:
        if node.right.is_limit_ordinal():
          # todo 1: record inside, record FS type
          try:
            old, recorder.rec_limit = recorder.rec_limit, 0
            return Ord(node.token,
                       node.left,
                       impl(node.right, n))
          finally:
            recorder.rec_limit = old

        else:
          if node.right == 0:
            return Ord(1)
          if node.right == 1:
            return node.left
          return Ord('+' if node.token.v == '*' else '*',
                      Ord.simplified(Ord(node.token, node.left, node.right.dec())),
                      node.left
                     )

      match ord.token.v:
        case '+':
          recorder.skip_next()
          new_pre = ord.left \
                    if rec_pre is None \
                    else Ord('+', rec_pre, ord.left)
          return Ord(ord.token,
                     ord.left,
                     impl(ord.right, n, new_pre))
        case '*':
          return impl(transform(ord), n, rec_pre)
        case '^':
          return impl(transform(ord), n, rec_pre)
        case _:
          assert 0, ord

    res = impl(self.ord, self.n)
    recorder.record(res, res=True)
    return res

  def calc_display(self, expected=None, test_only=False, show_steps=False):

    first = True

    def display(obj : FdmtSeq | Ord):
      if isinstance(obj, Ord):
        # todo 1: for w^(w[3])
        obj = FdmtSeq(obj, self.n)
      #   return obj.to_latex()
      nonlocal first
      ret = obj.to_latex(always_show_idx=first)
      first = False
      return ret

    def calc(fs: FdmtSeq, recorder):
      return FdmtSeq(fs.calc(recorder), self.n)

    return test_display(self, calc, display, expected,
                        Ord.fs_cal_limit_default, test_only, show_steps)


class FGH:
  ord: Ord
  x: int | FGH
  exp: int

  def __init__(self, ord, x, exp=1):
    self.ord = Ord.from_any(ord)
    self.x   = x
    self.exp = exp

  @staticmethod
  def seq(*args):
    assert len(args) > 1
    ret = args[-1]
    for ord in args[-2::-1]:
      ret = FGH(Ord.from_str(ord), ret)
    return ret

  def __eq__(self, other):
    if isinstance(self.x, int) != isinstance(other.x, int):
      return False
    return self.ord == other.ord and self.x == other.x and self.exp == other.exp

  def exp_str(self):
    return "" if self.exp == 1 else f"^{{{self.exp}}}"

  def __str__(self):
    return f'f{self.exp_str()}({self.ord}, {self.x})'

  def to_latex(self):
    x_latex = self.x if isinstance(self.x, int) else self.x.to_latex()
    return f'f_{{{self.ord.to_latex()}}}{self.exp_str()}({x_latex})'

  # return (succ, res)
  # limit_f2: max n for f2(n) to be eval
  def expand_once(self, digit_limit=1000) -> Tuple[bool, FGH | int]:
    if isinstance(self.x, FGH):
      succ, x2 = self.x.expand_once()
      return succ, FGH(self.ord, x2, self.exp)
    if self.ord.is_limit_ordinal():
      return True, FGH(self.ord.fs_at(
        self.x, Recorder(0, Ord.fs_cal_limit_default)), self.x)
    elif self.ord == Ord('0'):
      return True, self.x + self.exp
    elif self.ord == Ord('1'):
      if self.exp > digit_limit * 3:
        return False, self
      return True, self.x * (2 ** self.exp)
    elif self.ord == Ord('2'):
      if self.x > digit_limit * 3:
        return False, self
      new_x = (2 ** self.x) * self.x
      return True, new_x if self.exp == 1 else FGH(2, new_x, self.exp - 1)
    else:  # any ord >= 3
      dec1 = self.ord.dec()
      return True, FGH(dec1, FGH(dec1, self.x), self.x - 1)

  def expand_once_display(self, expected=None, test_only=False):
    succ, res = self.expand_once()
    if expected is not None:
      assert expected == res, f'{expected} != {res}'
    if test_only:
      return None
    res_str = res.to_latex() if isinstance(res, FGH) else str(res)
    maybe_unfinished = isinstance(res, FGH) and succ
    return f'{self.to_latex()}={res_str}' + \
           ('=...' if maybe_unfinished else '')

  def expand(self, recorder : Recorder):
    ret : FGH | int = self

    recorder.record(ret)

    for _ in range(recorder.cal_limit):
      succ, ret = cast(FGH, ret).expand_once()
      if succ:
        recorder.record(ret, res=True)
      if not succ or not isinstance(ret, FGH):
        return ret
    return ret

  def expand_display(self, expected=None, limit=100,
    test_only=False, show_steps=False):

    def fgh_to_latex(fgh : FGH | int):
      if isinstance(fgh, FGH):
        return fgh.to_latex()
      return str(fgh)

    def calc(fgh : FGH, recorder):
      return fgh.expand(recorder)

    return test_display(self, calc, fgh_to_latex, expected,
                        limit, test_only, show_steps)
