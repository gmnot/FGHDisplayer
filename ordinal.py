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

# Ordinal
class Ord:
  value: str  # operator +,*,^ ; natural number str; w,e ; Veblen
  left : Ord
  right: Ord

  ord_mappings : Dict[str, str] = {
    'w': r'\omega',
    'e': r'\varepsilon_{0}',
    'x': r'\xi_{0}',
    'h': r'\eta_{0}',
  }

  def __init__(self, value, left=None, right=None):
    self.value = value
    self.left  = left
    self.right = right

  def replace(self, other: Ord):
    self.value = other.value
    self.left  = other.left
    self.right = other.right

  def __eq__(self, other):
    # todo: consider associative laws
    return self.value == other.value and \
           self.left.__eq__(other.left) and \
           self.right.__eq__(other.right)

  def to_infix(self):
    if self.is_atomic():
      return str(self.value)
    return f"({self.left.to_infix()} {self.value} {self.right.to_infix()})"

  def __str__(self):
    return self.to_infix()

  def is_atomic(self):
    assert (self.left is None) == (self.right is None), self
    return self.left is None

  def is_valid(self):
    return self.is_atomic() or \
           self.left.is_valid() and self.right.is_valid()

  def is_natural(self):
    return all('0' <= c <= '9' for c in self.value)

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

    def is_operand(token):
      return token.isdigit() or token in Ord.ord_mappings.keys()

    def to_postfix(tokens):
      """
      Convert the infix expression tokens to postfix (RPN) using Shunting Yard algorithm.
      """
      out = []
      stack = []
      for token in tokens:
        if is_operand(token):
          out.append(token)
        elif token == "(":
          stack.append(token)
        elif token == ")":
          while stack and stack[-1] != "(":
            out.append(stack.pop())
          stack.pop()  # pop the '('
        else:  # Operator
          while (stack and precedence(stack[-1]) >= precedence(token)):
            out.append(stack.pop())
          stack.append(token)

      while stack:
        out.append(stack.pop())
      return out

    def build_tree(postfix) -> Ord:
      """
      Build a binary tree from the postfix expression.
      """
      stack = []
      for token in postfix:
        if is_operand(token):
          stack.append(Ord(token))
        else:  # Operator
          right = stack.pop()
          left = stack.pop()
          stack.append(Ord(token, left, right))

      kraise(len(stack) != 1,
             f"Can't read ordinal from {expression}: " +
             f"len(stack) is {len(stack)}: " + ' '.join(str(ord) for ord in stack)
             if debug_mode
             else f"{len(stack)} terms found. If you believe your input is valid" +
                  contact_request)
      return stack[0]

    try:
      tokens = re.findall(r'\d+|[+*^()]|' + '|'.join(Ord.ord_mappings.keys()), expression)
      # Convert infix to postfix (RPN)
      postfix = to_postfix(tokens)
      return build_tree(postfix)
    except KnownError as e:
      raise e
    except Exception as e:
      raise KnownError(f"Can't read ordinal from {expression}: " +
                       str(e) if debug_mode else
                       f"If you believe your input is valid, {contact_request}.")

  def ord_to_latex(self):
    """
    basic ord symbol to latex.
    1,2,3,...
    ord_mappings.keys(),
    Veblen
    """
    if self.is_natural():
      return self.value
    if self.value in Ord.ord_mappings.keys():
      return Ord.ord_mappings[self.value]
    assert 0, self.value

  def op_to_latex(self):
    match self.value:
      case '+':
        return '+'
      case '*':
        return r'\cdot'
      case '^':
        return '^'
      case _:
        assert 0, self.value

  def val_to_latex(self):
    if self.is_atomic():
      return self.ord_to_latex()
    else:
      return self.op_to_latex()

  def to_latex(self):
    if self.is_atomic():
      return self.ord_to_latex()

    def parentheses_on_demand(node: Ord, out_op: str):
      if node.is_atomic() or \
         (out_op == '*' and node.value == '^'):
        return node.to_latex()
      return '{(' + node.to_latex() + ')}'

    match self.value:
      case '+':
        return self.left.to_latex() + self.op_to_latex() + self.right.to_latex()
      case '*':
        return parentheses_on_demand(self.left, '*') + \
               self.op_to_latex() + \
               parentheses_on_demand(self.right, '*')
      case '^':
        return parentheses_on_demand(self.left, '^') + \
               self.op_to_latex() + \
               '{' + self.right.to_latex() + '}'

  def is_limit_ordinal(self):
    if self.is_atomic():
      return not self.is_natural()
    if self.value == '+':
      return self.right.is_limit_ordinal()
    # a * b and a ^ b is surely limit ordinal
    assert not self.left.is_natural(), self
    return True

  def dec(self) -> Ord:
    if self.is_atomic():
      assert self.is_natural()
      i = int(self.value)
      assert i > 0, self
      return Ord(str(i - 1))
    else:
      assert self.value == '+', self
      if self.right == Ord('1'):
        return self.left
      return Ord(self.value, self.left, self.right.dec())

  def simplify(self):
    if self.value in '*^' and self.right == Ord('1'):
      self.value, self.left, self.right = \
        self.left.value, self.left.left, self.left.right

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

    def impl(ord, n, record : RecType = RecType.FALSE, rec_pre=None) -> Ord:
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
        assert ord.value in Ord.ord_mappings.keys(), ord
        match ord.value:
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
          return Ord(node.value,
                     node.left,
                     impl(node.right, n))
        else:
          assert node.right != Ord('0'), ord
          if node.right == Ord('1'):
            return node.left
          return Ord('+' if node.value == '*' else '*',
                      Ord.simplified(Ord(node.value, node.left, node.right.dec())),
                      node.left
                     )

      match ord.value:
        case '+':
          new_rec = RecType.SKIP if record == RecType.TRUE else RecType.FALSE
          new_pre = ord.left \
                    if rec_pre is None \
                    else Ord('+', rec_pre, ord.left)
          return Ord(ord.value,
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

  def fundamental_sequence_display(self, n : str, expected=None, show_steps=False):
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

  def __init__(self, ord: Ord, x):
    self.ord = ord
    self.x   = x

  @staticmethod
  def seq(*args):
    assert len(args) > 1
    ret = args[-1]
    for ord in args[-2::-1]:
      ret = FGH(Ord.from_str(ord), ret)
    return ret

  def __eq__(self, other):
    return self.ord == other.ord and self.x == other.x

  def __str__(self):
    return f'f({self.ord}, {self.x})'

  def to_latex(self):
    x_latex = self.x if isinstance(self.x, int) else self.x.to_latex()
    return f'f_{{{self.ord.to_latex()}}}({x_latex})'

  # return (succ, res)
  def expand_once(self, limit=1000):
    if isinstance(self.x, FGH):
      succ, x2 = self.x.expand_once()
      return succ, FGH(self.ord, x2)
    if self.ord.is_limit_ordinal():
      return True, FGH(self.ord.fundamental_sequence_at(self.x)[0], self.x)
    elif self.ord == Ord('0'):
      return True, self.x + 1
    elif self.ord == Ord('1'):
      return True, self.x * 2
    elif self.ord == Ord('2'):
      if self.x > limit:
        return False, self
      return True, (2 ** self.x) * self.x
    else:
      if self.x > limit:
        return False, self
      dec1 = self.ord.dec()
      ret = self.x
      for _ in range(self.x):
        ret = FGH(dec1, ret)
      return True, ret

  def expand_once_display(self, expected=None):
    succ, ex1 = self.expand_once()
    if expected is not None:
      assert expected == ex1, f'{expected} != {ex1}'
    if isinstance(ex1, FGH):
      return f'{self.to_latex()}={ex1.to_latex()}'
    return f'{self.to_latex()}={ex1}'

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
