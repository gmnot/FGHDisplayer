from __future__ import annotations
import re

class Node:
  value: str
  left : Node
  right: Node

  def __init__(self, value, left=None, right=None):
    """
    Node represents an operator or operand in the expression tree.
    :param value: The operator (+, *, ^) or operand (natural number or 'w')
    :param left: Left subtree (Expression to the left of the operator)
    :param right: Right subtree (Expression to the right of the operator)
    """
    self.value = value
    self.left = left
    self.right = right

  def replace(self, other: Node):
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
    assert (self.left is None) == (self.right is None)
    return self.left is None

  def is_natural(self):
    return all('0' <= c <= '9' for c in self.value)

  @staticmethod
  def from_str(expression: str):
    """
    Read an infix expression and construct the corresponding binary tree.
    """
    def precedence(op):
      if op == "+":
        return 1
      if op == "*":
        return 2
      if op == "^":
        return 3
      return 0

    def to_postfix(tokens):
      """
      Convert the infix expression tokens to postfix (RPN) using Shunting Yard algorithm.
      """
      out = []
      stack = []
      for token in tokens:
        if token.isdigit() or token == 'w':  # Operand (natural number or 'w')
          out.append(token)
        elif token == "(":  # Left Parenthesis
          stack.append(token)
        elif token == ")":  # Right Parenthesis
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

    def build_tree(postfix) -> Node:
      """
      Build a binary tree from the postfix expression.
      """
      stack = []
      for token in postfix:
        if token.isdigit() or token == 'w':  # Operand
          stack.append(Node(token))
        else:  # Operator
          right = stack.pop()
          left = stack.pop()
          stack.append(Node(token, left, right))
      return stack[0]

    # Tokenize the expression
    tokens = re.findall(r'\d+|w|[+*^()]', expression)

    # Convert infix to postfix (RPN)
    postfix = to_postfix(tokens)

    # Build and return the tree from the postfix expression
    return build_tree(postfix)

  def ord_to_latex(self):
    """
    basic ord symbol to latex.
    such as 1,2,3..., w, \varepsilon, ...
    """
    if self.is_natural():
      return self.value
    assert self.value == 'w'
    return r'\omega'

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
    return '{' + self.left.to_latex() + '}' + \
           self.op_to_latex() + \
           '{' + self.right.to_latex() + '}'

  def dec(self) -> Node:
    assert self.is_atomic() and self.is_natural()
    i = int(self.value)
    assert i > 0, self
    return Node(str(i - 1))

  def simplify(self):
    if self.value in '*^' and self.right == Node('1'):
      self.value, self.left, self.right = \
        self.left.value, self.left.left, self.left.right

  @staticmethod
  def simplified(node: Node):
    node.simplify()
    return node

  def fundamental_sequence_at(self, n) -> Node:
    if self.is_atomic():
      assert not self.is_natural(), self
      assert self.value == 'w'
      return Node(str(n))

    def reduce(node: Node) -> Node:
      if node.right.is_natural():
        assert node.right != Node('0'), node
        if node.right == Node('1'):
          return node.left
        return Node('+' if node.value == '*' else '*',
                    Node.simplified(Node(node.value, node.left, node.right.dec())),
                    node.left
                    )
      else:
        return Node(node.value,
                    node.left,
                    node.right.fundamental_sequence_at(n))

    match self.value:
      case '+':
        return Node(self.value, self.left, self.right.fundamental_sequence_at(n))
      case '*':
        return reduce(self).fundamental_sequence_at(n)
      case '^':
        return reduce(self).fundamental_sequence_at(n)
      case _:
        assert 0, self
class Ordinal:
  root : Node

  def __init__(self, root=None):
    """
    Expression represents the entire mathematical expression as a binary tree.
    :param root: The root node of the expression (main operator)
    """
    self.root = root

  def __str__(self):
    """
    Converts the expression tree to an infix string representation (with parentheses).
    """
    return "" if self.root is None else self.root.__str__()

  def __eq__(self, other):
    return self.root == other.root

  @staticmethod
  def from_str(expression: str):
    return Ordinal(Node.from_str(expression))

  def to_latex(self):
    return self.root.to_latex()

  def fundamental_sequence_at(self, n) -> Node:
    assert self.root is not None
    return self.root.fundamental_sequence_at(n)

  def fundamental_sequence_display(self, n, expected=None):
    fs = self.fundamental_sequence_at(n)
    if expected is not None:
      assert fs == expected, f'{str(fs)} != {str(expected)}'
    return f'{self.to_latex()}[{n}]={fs.to_latex()}'

class FGH:
  ord: Ordinal
  x: int | FGH

  def __init__(self, ord: Ordinal, x):
    self.ord = ord
    self.x   = x

  def __eq__(self, other):
    return self.ord == other.ord and self.x == other.x

  def __str__(self):
    return f'f({self.ord}, {self.x})'

  def to_latex(self):
    x_latex = self.x if isinstance(self.x, int) else self.x.to_latex()
    return f'f_{{{self.ord.to_latex()}}}({x_latex})'

  def expand_once(self):
    if isinstance(self.x, FGH):  # todo: expand small
      return self
    return FGH(Ordinal(self.ord.fundamental_sequence_at(self.x)), self.x)

  def expand_once_display(self, expected=None):
    ex1 = self.expand_once()
    if expected is not None:
      assert expected == ex1, f'{expected} != {ex1}'
    return f'{self.to_latex()}={ex1.to_latex()}'
