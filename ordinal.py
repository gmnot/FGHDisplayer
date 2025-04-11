from __future__ import annotations
from copy import deepcopy
from enum import Enum
from typing import List, Dict, Tuple
import re

from html_utils import contact_request

"""
todo:
- copy examples
- f_2^2(3) to save space
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
    # todo 1: is int
    return isinstance(self.v, int) or self.v.isdigit()

  def to_latex(self):
    if self.is_natural():
      return str(self.v)
    if self.v in Token.latex_maps.keys():
      return Token.latex_maps[self.v]
    assert 0, self

# Ordinal
class Ord:
  token: Token  # operator +,*,^ ; natural number str; w,e ; Veblen
  left : Ord
  right: Ord

  def __init__(self, token, left=None, right=None):
    self.token = Token(token)
    self.left  = left
    self.right = right

  def __eq__(self, other):
    # todo: consider associative laws
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
      # todo 1
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
      assert self.is_natural()
      i = int(self.token.v)
      assert i > 0, self
      return Ord(str(i - 1))
    else:
      assert self.token.v == '+', self
      if self.right == Ord('1'):
        return self.left
      return Ord(self.token, self.left, self.right.dec())

  def simplify(self):
    if self.token.v in '*^' and self.right == Ord('1'):
      self.token, self.left, self.right = \
        self.left.token, self.left.left, self.left.right

  @staticmethod
  def simplified(node: Ord):
    node.simplify()
    return node

  def fundamental_sequence_at(self, n, record_steps=False, record_limit=12) \
    -> Tuple[Ord, List[Ord]]:

    steps : List[Ord] = []

    class RecType(Enum):
      FALSE = 0
      TRUE  = 1
      SKIP  = 2  # skip just one, and true for following

    def impl(ord : Ord, n, record : RecType = RecType.FALSE, rec_pre=None) \
      -> Ord:

      if record == RecType.TRUE and len(steps) < record_limit:
        if rec_pre is None:
          steps.append(ord)
        else:
          assert rec_pre.is_valid()
          steps.append(Ord('+', rec_pre, ord))

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

    res = impl(self, n, RecType.TRUE if record_steps else RecType.FALSE)
    return res, steps

  def fundamental_sequence_display(self, n : int, expected=None, show_steps=False):
    fs, steps = self.fundamental_sequence_at(n, record_steps=show_steps)
    if expected is not None:
      assert fs == expected, f'{str(fs)} != {str(expected)}'

    if not show_steps:
      return f'{self.to_latex()}[{n}]={fs.to_latex()}'

    ret = r' \begin{align*}' + '\n'
    ret += f'{steps[0].to_latex()}[{n}]'
    for ord in steps[1:] + [fs]:
      ret += f'  &= {ord.to_latex()}[{n}] ' + r'\\' + '\n'
    ret += r'\end{align*} ' + '\n'
    return ret

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
    return "" if self.exp == 1 else f"^{self.exp}"

  def __str__(self):
    return f'f{self.exp_str()}({self.ord}, {self.x})'

  def to_latex(self):
    x_latex = self.x if isinstance(self.x, int) else self.x.to_latex()
    return f'f_{{{self.ord.to_latex()}}}{self.exp_str()}({x_latex})'

  # return (succ, res)
  def expand_once(self, limit=1000):
    if isinstance(self.x, FGH):
      succ, x2 = self.x.expand_once()
      return succ, FGH(self.ord, x2, self.exp)
    if self.ord.is_limit_ordinal():
      return True, FGH(self.ord.fundamental_sequence_at(self.x)[0], self.x)
    elif self.ord == Ord('0'):
      return True, self.x + self.exp
    elif self.ord == Ord('1'):
      return True, self.x * (2 ** self.exp)
    elif self.ord == Ord('2'):
      # ! check exp
      if self.x > limit:
        return False, self
      new_x = (2 ** self.x) * self.x
      return True, new_x if self.exp == 1 else FGH(2, new_x, self.exp - 1)
    else:
      if self.x > limit:
        return False, self
      dec1 = self.ord.dec()
      return True, FGH(dec1, FGH(dec1, self.x), self.x - 1)
      ret = self.x
      for _ in range(self.x):
        ret = FGH(dec1, ret)
      return True, ret

  def expand_once_display(self, expected=None):
    succ, res = self.expand_once()
    if expected is not None:
      assert expected == res, f'{expected} != {res}'
    res_str = res.to_latex() if isinstance(res, FGH) else str(res)
    maybe_unfinished = isinstance(res, FGH) and succ
    return f'{self.to_latex()}={res_str}' + \
           ('=...' if maybe_unfinished else '')

  def expand(self, limit=1000):
    ret = self
    for _ in range(limit):
      succ, ret = ret.expand_once(limit)
      if not succ or not isinstance(ret, FGH):
        return ret
    return ret

  def expand_display(self, expected=None, limit=1000):
    ex1 = self.expand(limit=limit)
    if expected is not None:
      assert expected == ex1, f'{expected} != {ex1}'
    if isinstance(ex1, FGH):
      return f'{self.to_latex()}={ex1.to_latex()}'
    return f'{self.to_latex()}={ex1}'
