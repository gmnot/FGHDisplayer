from __future__ import annotations
from copy import deepcopy
from enum import Enum
from typing import List, Dict, Tuple, cast
import re

from html_utils import contact_request

"""
todo:
"""

debug_mode = False
def ord_set_debug_mode(new : bool):
  global debug_mode
  debug_mode = new

class KnownError(Exception):
  @staticmethod
  def raise_if(cond, msg):
    if cond:
      raise KnownError(msg)

def kraise(*args):
  return KnownError.raise_if(*args)

class Veblen:
  param: List[Ord]

  def __init__(self, *args: Ord):
    assert len(args) > 0
    self.param = [o for o in args[::-1]]

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
        if v.isdigit():
          self.v = int(v)
        else:
          self.v = v
      case _:
        assert 0, v

  def __eq__(self, other):
    return type(self.v) == type(other.v) and self.v == other.v

  def __str__(self):
    return str(self.v)

  def is_natural(self):
    return isinstance(self.v, int)

  def get_natrual(self) -> int:
    assert self.is_natural()
    return cast(int, self.v)

  def to_latex(self):
    if self.is_natural():
      return str(self.v)
    if self.v in Token.latex_maps.keys():
      return Token.latex_maps[self.v]
    assert 0, self

class Record:
  rec_limit : int
  data      : List  # allow extra elem for result
  full      : bool
  n_discard : int

  def __init__(self, rec_limit):
    assert rec_limit >= 2
    self.rec_limit = rec_limit
    self.data      = []
    self.full      = False
    self.n_discard = 0

  def no_mid_steps(self):
    return self.rec_limit == 2

  def record(self, entry, res=False):
    if self.full:
      self.n_discard += 1
      if res:  # force replace
        self.data[-1] = entry
    else:
      self.data.append(entry)
      assert len(self.data) <= self.rec_limit
      if len(self.data) == self.rec_limit - 1:  # save 1 for result
        self.full = True

  def to_latex(self, entry_to_latex):
    ret = r' \begin{align*}' + '\n'
    ret += entry_to_latex(self.data[0])
    for ord in self.data[1:-1]:
      ret += f'  &= {entry_to_latex(ord)} \\\\\n'
    if self.n_discard > 0:
      ret += r'  &\phantom{=} \vdots \quad \raisebox{0.2em}{\text{' + \
             f'after {self.n_discard} more steps' r'}} \\' + '\n'
    ret += f'  &= {entry_to_latex(self.data[-1])} \\\\\n'
    ret += r'\end{align*} ' + '\n'
    return ret

def test_display(obj, f_calc, f_display, expected=None,
                 limit=100, test_only=False , show_steps=False):
  recorder = Record(15) if show_steps else None
  res = f_calc(obj, limit, recorder)

  if expected is not None:
    assert res == expected, f'{res} != {expected}'

  if test_only:
    return None

  if not show_steps:
    return f'{f_display(obj)}={f_display(res)}'

  assert recorder is not None
  return recorder.to_latex(f_display)

# Ordinal
class Ord:
  token: Token  # operator +,*,^ ; natural number str; w,e ; Veblen
  left : Ord
  right: Ord

  def __init__(self, token, left=None, right=None):
    self.token = Token(token)
    self.left  = left
    self.right = right

  def rotate(self) -> None:
    if self.is_atomic():
      return
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
    self.rotate()
    other.rotate()
    return self.token == other.token and \
           self.left.__eq__(other.left) and \
           self.right.__eq__(other.right)

  def to_infix(self):
    if self.is_atomic():
      return str(self.token)
    return f"({self.left.to_infix()} {self.token} {self.right.to_infix()})"

  def __str__(self):
    return self.to_infix()

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

    def is_operand(tok):
      return tok.isdigit() or tok in Token.ord_maps

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
      tokens = re.findall(r'\d+|[+*^()]|' + '|'.join(Token.ord_maps), expression)
      # Convert infix to postfix (RPN)
      postfix = to_postfix(tokens)
      return build_tree(postfix)
    except KnownError as e:
      raise e
    except Exception as e:
      raise KnownError(f"Can't read ordinal from {expression}: " +
                       str(e) if debug_mode else
                       f"If you believe your input is valid, {contact_request}.")

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

  def fundamental_sequence_at(self, n, rec : Record | None = None) -> Ord:
    class RecType(Enum):
      FALSE = 0
      TRUE  = 1
      SKIP  = 2  # skip just one, and true for following

    def impl(ord : Ord, n, record : RecType = RecType.FALSE, rec_pre=None) \
      -> Ord:

      if record == RecType.TRUE:
        assert rec is not None
        if rec_pre is None:
          rec.record(ord)
        else:
          assert rec_pre.is_valid()
          rec.record(Ord('+', rec_pre, ord))

      def update(rec):  # SKIP -> TRUE, other->other
        return RecType.TRUE if rec != RecType.FALSE else RecType.FALSE

      if ord.is_atomic():
        if ord.is_natural():
          return ord
        match ord.token.v:
          case 'w':
            return Ord(str(n))
          case 'e':
            return impl(Ord.from_str('w^('*(n-1) + 'w' + ')'*(n-1)), n,
                        update(record), rec_pre)
          case _:
            assert 0, f'{ord} @ {n}'

      # transform w*2 and w^2 so they can be indexed
      def transform(node: Ord) -> Ord:
        if node.right.is_limit_ordinal():
          return Ord(node.token,
                     node.left,
                     impl(node.right, n))
        else:
          assert node.right != Ord('0'), ord
          if node.right == Ord('1'):
            return node.left
          return Ord('+' if node.token.v == '*' else '*',
                      Ord.simplified(Ord(node.token, node.left, node.right.dec())),
                      node.left
                     )

      match ord.token.v:
        case '+':
          new_rec = RecType.SKIP if record == RecType.TRUE else RecType.FALSE
          new_pre = ord.left \
                    if rec_pre is None \
                    else Ord('+', rec_pre, ord.left)
          return Ord(ord.token,
                     ord.left,
                     impl(ord.right, n, new_rec , new_pre))
        case '*':
          return impl(transform(ord), n, update(record), rec_pre)
        case '^':
          return impl(transform(ord), n, update(record), rec_pre)
        case _:
          assert 0, ord

    res = impl(self, n, RecType.TRUE if rec is not None else RecType.FALSE)
    if rec is not None:
      rec.record(res, res=True)
    return res

  def fundamental_sequence_display(self,
    n : int, expected=None, test_only=False, show_steps=False):

    first = True

    def ord_to_fs(ord : Ord):
      nonlocal first
      ret = f'{ord.to_latex()}'
      if first or ord.is_limit_ordinal():
        ret += f'[{n}]'
        first = False
      return ret

    def calc(ord, limit, recorder):
      return ord.fundamental_sequence_at(n, recorder)

    return test_display(self, calc, ord_to_fs, expected,
                        None, test_only, show_steps)

class FGH:
  ord: Ord
  x: int | FGH
  exp: int

  def __init__(self, ord, x, exp=1):
    if isinstance(ord, Ord):
      self.ord = ord
    elif isinstance(ord, str):
      self.ord = Ord.from_str(ord)
    elif isinstance(ord, int):
      self.ord = Ord.from_str(str(ord))
    else:
      assert 0, ord
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
      return True, FGH(self.ord.fundamental_sequence_at(self.x), self.x)
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

  def expand(self, limit=100, recorder : Record | None = None):
    ret : FGH | int = self
    if recorder:
      recorder.record(ret)

    for _ in range(limit):
      succ, ret = cast(FGH, ret).expand_once()
      if succ and recorder:
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

    def calc(fgh, limit, recorder):
      return fgh.expand(limit, recorder)

    return test_display(self, calc, fgh_to_latex, expected,
                        limit, test_only, show_steps)
