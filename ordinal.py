from __future__ import annotations
from copy import copy, deepcopy
from enum import Enum
from typing import Any, List, Dict, Tuple, cast
import re
import utils

from html_utils import contact_request

"""
todo:
"""

# * globals
debug_mode = False
debug_rotate = False
rotate_counter : int = 0

def ord_set_debug_mode(new : bool):
  global debug_mode
  debug_mode = new

def get_rotate_counter() -> int:
  global rotate_counter
  return rotate_counter

# * Exceptions
class KnownError(Exception):
  @classmethod
  def raise_if(cls, cond, msg):
    if cond:
      raise cls(msg)

def kraise(*args):
  return KnownError.raise_if(*args)

class WIPError(KnownError):
  pass

def strip_pre_post(pre: str, s: str, post: str) -> str:
  l1, l2 = len(pre), len(post)
  assert len(s) >= l1 + l2, f'{s} {pre} {post}'
  assert s.startswith(pre) and s.endswith(post), f'{s} {pre} {post}'
  return s[l1:-l2]

def to_latex(obj) -> str:
  match obj:
    case int():
      return str(obj)
    case str():
      return obj
    case _:
      return obj.to_latex()

class Veblen:
  param: List[Ord]  # v:       v(1, 0, 0)
                    # index:     0  1  2
  latex_force_veblen: bool  # force showing forms like v(0, 1) in latex

  def __init__(self, *args, latex_force_veblen=False):
    assert len(args) >= 2
    self.param = [Ord.from_any(o) for o in args]
    self.latex_force_veblen = latex_force_veblen

  @staticmethod
  def from_str(s_ : str, *, latex_force_veblen=False) -> Veblen:
    s = strip_pre_post('v(', s_, ')').strip()
    ords = (Ord.from_str(s, latex_force_veblen=latex_force_veblen) for s in s.split(','))
    return Veblen(*ords, latex_force_veblen=latex_force_veblen)

  def __eq__(self, other):
    # todo: consider complex Veblen
    return isinstance(other, Veblen) and \
           all(v == o for v, o in zip(self.param, other.param))

  def __add__(self, rhs):
    return Ord('+', self, rhs)

  def __str__(self):
    return 'v({})'.format(', '.join(
      (str(o) if o is not None else '.' for o in self.param)
    ))

  def __repr__(self):
    return self.__str__()

  def idxs_missing(self):
    return [i for i, val in enumerate(self.param) if val is None]

  def is_missing_one(self):
    return len(self.idxs_missing()) == 1

  def make_combined(self, other) -> Ord:
    idxs = self.idxs_missing()
    assert len(idxs) == 1, self
    idx = idxs[0]
    ret = Veblen(*self.param)
    ret.param[idx] = other
    return Ord(ret)

  def to_latex(self):
    if self.is_binary() and not self.latex_force_veblen:
      a = self.param[0].token.v
      if isinstance(a, int) and 0 <= a <= 3:
        ax = [r'\omega', r'\varepsilon', r'\xi', r'\eta'][a]
        gx = self.param[1].to_latex()
        if a == 0:
          return f'{ax}^{{{gx}}}'
        return f'{ax}_{{{gx}}}'

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

  def index(self, n : int, rec: Recorder) -> Tuple[bool, Ord | None, FdmtSeq]:

    WIPError.raise_if(not self.is_binary(),
                      f"WIP: multi-var Veblen will be available soon. {self}")
    ax = self.param[0]   # first non-zero term except last term. a or a+1
    gx = self.param[-1]  # last term, g or g+1

    def succ(nxt, remain=None):
      return (True, remain, FdmtSeq(nxt, n))

    def succ_v(v: Tuple | Ord, nxt, *, n_nxt=n):
      return (True,
              (Ord(Veblen(*v)) if isinstance(v, tuple) else Ord(v)),
              FdmtSeq(nxt, n_nxt))

    if gx == 0 and n == 0:  # R4: v(a, 0)[0] = 0
      return succ(Ord(0))
    if ax == 0:  # R2: v(0,g) = w^g (for any g)
      # rec.skip_next()
      return succ(Ord('^', 'w', gx))
    if gx.is_limit_ordinal():  # R3, g is LO
      return succ_v((ax, None), gx)
    if ax.is_limit_ordinal():  # R8-9 v(a, .)
      if gx == 0:  # R8 v(a, 0)
        # todo 1: record
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
          # todo 1: show dec
          return succ(Veblen(ax, gx.dec()) + 1)
        else:  # R7 v(a+1,g+1)[n+1]: g -> v(a, g)
          # e.g. e2 = {e1+1, w^(...), w^w^(...), }
          return succ_v((a, None), self, n_nxt=n-1)

# number, w, e, operators, Veblen
class Token:
  v: int | str | Veblen | FdmtSeq
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

  def __init__(self, v, *, latex_force_veblen=False):
    match v:
      case Token():
        self.v = v.v
      case int() | Veblen() | FdmtSeq():
        self.v = v
      case str():
        assert len(v) > 0
        if v.isdigit():
          self.v = int(v)
        elif v[0] == 'v':
          self.v = Veblen.from_str(v, latex_force_veblen=latex_force_veblen)
        elif v in 'exh':
          self.v = Veblen('exh'.index(v)+1, 0)
        elif v in 'w+*^':
          self.v = v
        else:
          assert 0, self.v
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
    match self.v:
      case int():
        return str(self.v)
      case Veblen() | FdmtSeq():
        return self.v.to_latex()
      case str():
        assert self.v in Token.latex_maps.keys()
        return Token.latex_maps[self.v]
      case _:
        assert 0, self

class Recorder:
  rec_limit : int
  cal_limit : int
  data      : List  # allow extra elem for result
  n_discard : int
  n_pre_discard : int  # discarded before reach limit
  will_skip_next : bool  # skip the next record

  until     : Any
  until_met : bool

  def __init__(self, rec_limit, cal_limit, until=None):
    assert rec_limit >= 0
    self.rec_limit = rec_limit
    self.cal_limit = cal_limit
    self.data      = []
    self.n_discard = 0
    self.n_pre_discard = 0
    self.will_skip_next = False
    self.until = until
    self.until_met = False

  def __repr__(self):
    return f'{self.data}'

  def active(self):
    return self.rec_limit != 0

  def cnt(self) -> int:
    return len(self.data)

  def tot_discard(self):
    return self.n_discard + self.n_pre_discard

  def cal_limit_reached(self):
    return len(self.data) + self.tot_discard() >= self.cal_limit

  def no_mid_steps(self):
    return self.rec_limit == 2

  # return: True if cal_limit_reached
  def record(self, entry, res=False) -> bool:
    try:
      if not self.active():
        return False
      if self.will_skip_next:
        pass
      elif self.cnt() >= self.rec_limit:
        self.n_discard += 1
        if res:  # force replace
          self.data[-1] = entry
      else:
        self.data.append(entry)

      return self.inc_discard_check_limit()
    finally:
      self.will_skip_next = False

  def record_fs(self, pres : List[Ord], curr : Ord, res=False) -> bool:
    ord_list = pres + [curr]
    return self.record(Ord.combine_list(ord_list), res)

  def get_result(self):
    assert len(self.data) > 0
    return self.data[-1]

  # return True if cal_limit reached
  def inc_discard_check_limit(self) -> bool:
    if self.cnt() >= self.rec_limit:
      self.n_discard += 1
    else:
      self.n_pre_discard += 1
    return self.cal_limit_reached()

  def skip_next(self):
    self.will_skip_next = True

  # todo 1: one line mode; based on number of terms
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

# Binary Tree Node
class Node:
  token: Token  # operator +,*,^ ; natural number str; w,e ; Veblen
  left : Ord
  right: Ord

  def __init__(self, token, left=None, right=None, *, latex_force_veblen=False):
    self.token = Token(token, latex_force_veblen=latex_force_veblen)
    self.left  = Ord.from_any(left, latex_force_veblen=latex_force_veblen)
    self.right = Ord.from_any(right, latex_force_veblen=latex_force_veblen)

  def n_child(self):
    return (self.left is not None) + (self.right is not None)

  # give left or right subtree and a copy of remain to func
  def recurse_node(self, attr, func):
    sub_node = getattr(self, attr)
    remain   = copy(self)
    setattr(remain, attr, None)
    return func(sub_node, remain)

# Ordinal
class Ord(Node):
  rotate_at_init = True  # retire the other mode

  # * structual
  def __init__(self, token, left=None, right=None, *, latex_force_veblen=False):
    super().__init__(token, left, right, latex_force_veblen=latex_force_veblen)
    if Ord.rotate_at_init:
      self.rotate()

  # left-hand tree join elems with op
  @classmethod
  def from_list(cls, op: str, li: List[Ord]) -> Ord:
    res = li[-1]
    for t in li[-2::-1]:
      res = Ord(op, t, res)
    return res

  @utils.trace_calls(enabled=debug_rotate)
  def rotate_op(self, op: str, to_right) -> None:
    # for example, if to right:
    #          +                    +
    #        /   \                 / \
    #      +      +    ===>      ll    +
    #     / \    / \                  / \
    #   ...  ....   ...              rl  +
    #                                   / \
    #                                 ... ...

    def should_recur(node):
        return not node.is_atomic() and node.token == op

    def can_rotate(node):
      return should_recur(node) and \
        (to_right     and node.left.token  == op or
         not to_right and node.right.token == op)

    terms = []

    def gather(node: Ord):
      if not should_recur(node):
        if not Ord.rotate_at_init:
          node.rotate()
        terms.append(node)
      else:  # left + right
        gather(node.left)
        gather(node.right)

    if not can_rotate(self):
      if not Ord.rotate_at_init:
        self.left.rotate()
        self.right.rotate()
      return

    if Ord.rotate_at_init:
      use_simple = True  # idk why, but this records more rotated terms
      if use_simple:
        if to_right:
          terms = [self.left.left, self.left.right, self.right]
        else:
          terms = [self.left, self.right.left, self.right.right]
      else:
        gather(self)
    else:
      gather(self)
    global rotate_counter
    rotate_counter += len(terms)

    if to_right:
      res = Ord.from_list(op, terms)
      self.__dict__.update(res.__dict__)
    else:
      res = terms[0]
      for t in terms[1:]:
        res = Ord(op, res, t)
      self.__dict__.update(res.__dict__)

  @utils.track_total_time()
  def rotate(self) -> None:
    if self.is_atomic():
      return
    self.rotate_op('+', to_right=False)
    self.rotate_op('*', to_right=False)

  @staticmethod
  def rotated(ord : Ord) -> Ord:
    if ord is None:
      return None
    ord.rotate()
    return ord

  @classmethod
  def struct_eq(cls, a : Ord | None, b : Ord | None) -> bool:
    if a is None:
      return b is None
    if b is None:
      return False
    return a.token == b.token and \
           cls.struct_eq(a.left, b.left) and \
           cls.struct_eq(a.right, b.right)

  def __eq__(self, other_):
    other = Ord.from_any(other_)
    if self.token != other.token:
      return False
    if not Ord.rotate_at_init:
      self.rotate()
      other.rotate()
    return Ord.struct_eq(self, other)

  def to_infix(self):
    if self.n_child() == 0:
      return str(self.token)
    ret  = self.left.to_infix() if self.left else '.'
    ret += f'{self.token}'
    ret += self.right.to_infix() if self.right else '.'
    return f'({ret})'

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

  def combine_with(self, other: Ord):
    if self.left is None:
      assert self.right is not None
      self.left = other
    assert self.right is None
    self.right = other

  # search and combine <other> to missing node of self/subtree.
  # child_only: no recurse, fail if n_child != 1
  # return: None means failed, all full
  # note: must copy from the top. self can be changed later,
  #       like adding more half node. Returned Ord need to stay the same
  def make_combined(self, other: Ord, child_only=False) -> Ord | None:
    if self.n_child() == 0:
      if isinstance(self.token.v, Veblen) and self.token.v.is_missing_one():
        return self.token.v.make_combined(other)
      return None
    if self.left is None:
      assert self.right is not None
      return Ord(self.token, other, self.right)
    if self.right is None:
      return Ord(self.token, self.left,  other)
    if child_only:
      return None
    l = self.left.make_combined(other)
    if l is not None:
      return Ord(self.token, l, self.right)
    r = self.right.make_combined(other)
    if r is not None:
      return Ord(self.token, self.left, r)
    return None

  @classmethod
  def combine_list(cls, list: List[Ord]) -> Ord:
    assert len(list) != 0
    ret = list[-1]
    for o in list[-2::-1]:
      ret = utils.not_none(o.make_combined(ret))
    return ret

  # * named c'tors
  @staticmethod
  def from_str(expression: str, *, latex_force_veblen=False):
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
          stack.append(Ord(tok, latex_force_veblen=latex_force_veblen))
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
    except RecursionError as e:
      raise e
    except KnownError as e:
      raise e
    except Exception as e:
      raise KnownError(f"Can't read ordinal from {expression}: " +
                       str(e) if debug_mode else
                       f"If you believe your input is valid, {contact_request}.")

  @staticmethod
  def from_any(expression, *, latex_force_veblen=False) -> Ord:
    match expression:
      case None:
        return cast(Ord, None)
      case Ord():
        return expression
      case str():
        return Ord.from_str(expression, latex_force_veblen=latex_force_veblen)
      case int():
        return Ord.from_str(str(expression))
      case Veblen() | FdmtSeq():
        return Ord(expression, latex_force_veblen=latex_force_veblen)
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

  # * math
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
      if self.right == 1:
        return self.left
      return Ord(self.token, self.left, self.right.dec())

  def simplify(self):
    if self.token.v in '*^' and self.right == 1:
      self.token, self.left, self.right = \
        self.left.token, self.left.left, self.left.right

  @staticmethod
  def simplified(node: Ord):
    node.simplify()
    return node

  fs_cal_limit_default = 500
  # just a shortcut for FdmtSeq Func
  def fs_at(self, n, recorder : Recorder) -> Ord:
    return FdmtSeq(self, n).calc(recorder)

# Fundamental Sequence
class FdmtSeq:
  ord: Ord
  n  : int

  def __init__(self, ord, n: int, *, latex_force_veblen=False):
    self.ord = Ord.from_any(ord, latex_force_veblen=latex_force_veblen)
    self.n   = n

  def __eq__(self, other):
    return self.ord == other.ord and self.n == other.n

  def __str__(self):
    return f'{self.ord}[{self.n}]'

  def __repr__(self):
    return self.__str__()

  def to_latex(self, always_show_idx=True):
    ret = f'{self.ord.to_latex()}'
    if always_show_idx or self.ord.is_limit_ordinal():
      ret += f'[{self.n}]'
    return ret

  cal_limit_default = 500
  # must return Ord.
  # If return FS, outside can't know if expand has completed
  # But with Ord, outside know it's done if Ord.token isn't FS
  @utils.validate_return_based_on_arg(
      'recorder', lambda ret, rec: not debug_mode or ret == rec.get_result())
  def calc(self, recorder : Recorder, *, until=None) -> Ord:

    def impl(fs: FdmtSeq) -> Tuple[bool, Ord | None, FdmtSeq]:

      ord = fs.ord
      ret_failed = (False, None, fs)

      # * note: sub_node first
      def succ(sub_node: Ord, remain: Ord | None = None):
        return (True, remain, FdmtSeq(sub_node, fs.n))

      if ord.is_atomic():
        if ord.is_natural():
          return ret_failed
        match ord.token.v:
          case Veblen():
            return ord.token.v.index(fs.n, recorder)
          case 'w':
            return succ(Ord(fs.n))
          case 'e':
            recorder.skip_next()
            return succ(Ord(Veblen(1, 0)))
          case _:
            assert 0, f'{fs}'

      match ord.token.v:
        case '+':
          recorder.skip_next()
          return ord.recurse_node("right", succ)
        case '*' | '^':
          if ord.right.is_limit_ordinal():
            # (w^a)[3] = w^(a[3])
            if ord.token == '*':  # (w*w)[3] = w*(w[3]) looks the same
              recorder.skip_next()
            return ord.recurse_node("right", succ)
          else:
            if ord.right == 0:
              return succ(Ord(1))
            if ord.right == 1:
              return succ(ord.left)
            # w^(a+1)[3] = w^a * w[3]
            # todo 2: dec not recorded. combine w/ is_limit_ordinal case then go inside
            return succ(Ord('+' if ord.token.v == '*' else '*',
                        Ord.simplified(Ord(ord.token, ord.left, ord.right.dec())),
                        ord.left))
        case _:
          assert 0, ord

    pre_stack : List[Ord] = []
    curr : FdmtSeq = self

    # record orignal FS, and every time eval success
    if recorder.record_fs([], Ord.from_any(curr), res=True):
      return recorder.get_result()
    while True:
      succ, pre, next = impl(curr)  # ! idx could change! like R5
      if succ:  # curr expands to next
        if pre is not None:
          pre_stack.append(pre)
        if recorder.record_fs(copy(pre_stack), Ord.from_any(next), res=True):
          return recorder.get_result()
        curr = next
      else:  # can't eval curr
        assert pre is None
        # try combine and continue
        adds_reversed : List[Ord] = []
        non_add = None
        while len(pre_stack) > 0:
          lhs = pre_stack.pop()
          if lhs.token == '+':
            assert lhs.right is None
            adds_reversed.append(lhs.left)
          else:
            non_add = lhs
            break
        # add all
        if len(adds_reversed) > 0:
          curr = FdmtSeq(Ord.from_list('+', adds_reversed[::-1] + [next.ord]), self.n)
        if non_add:
          # * when can't eval, restore with n of *self*
          # e.g. e[1] = w^e[0] = w^(0[0]) = (w^0)[1]
          curr = FdmtSeq(utils.not_none(non_add.make_combined(curr.ord)), self.n)
        else:  # a+b+c+... and last can't eval, end.
          assert curr.n == self.n
          recorder.record_fs(copy(pre_stack), curr.ord, res=True)
          return curr.ord

class FGH:
  ord: Ord
  x: int | FGH
  exp: int

  def __init__(self, ord, x, exp=1, *, latex_force_veblen=False):
    self.ord = Ord.from_any(ord, latex_force_veblen=latex_force_veblen)
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
        self.x, Recorder(1, Ord.fs_cal_limit_default)), self.x)
    elif self.ord == 0:
      return True, self.x + self.exp
    elif self.ord == 1:
      if self.exp > digit_limit * 3:
        return False, self
      return True, self.x * (2 ** self.exp)
    elif self.ord == 2:
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
      assert res == expected, f'{res} != {expected}'
    if test_only:
      return None
    res_str = res.to_latex() if isinstance(res, FGH) else str(res)
    maybe_unfinished = isinstance(res, FGH) and succ
    return f'{self.to_latex()}={res_str}' + \
           ('=...' if maybe_unfinished else '')

  cal_limit_default = 100
  def calc(self, recorder : Recorder, until=None):
    ret : FGH | int = self

    recorder.record(ret)

    for _ in range(recorder.cal_limit):
      succ, ret = cast(FGH, ret).expand_once()
      if succ:
        recorder.record(ret, res=True)
      if not succ or not isinstance(ret, FGH):
        return ret
    return ret
