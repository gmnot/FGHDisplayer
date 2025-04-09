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

  def is_atomic(self):
    assert (self.left is None) == (self.right is None)
    return self.left is None

  def to_infix(self):
    if self.is_atomic():
      return str(self.value)
    return f"({self.left.to_infix()} {self.value} {self.right.to_infix()})"

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

  def fundamental_sequence_at(self, n) -> Node:
    if self.is_atomic():
      if self.is_natural():
        return Node(self.value)
      assert self.value == 'w'
      return Node(str(n))
    match self.value:
      case '+':
        return Node(self.value, self.left, self.right.fundamental_sequence_at(n))
      # todo: add here
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
    return "" if self.root is None else self.root.to_infix()

  @staticmethod
  def from_str(expression: str):
    return Ordinal(Node.from_str(expression))

  def to_latex(self):
    return self.root.to_latex()

  def fundamental_sequence_at(self, n) -> Node:
    assert self.root is not None
    return self.root.fundamental_sequence_at(n)

  def fundamental_sequence_display(self, n):
    fs = self.fundamental_sequence_at(n)
    return f'{self.to_latex()}[{n}]={fs.to_latex()}'

latex_html_headers = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>FGH </title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, { delimiters: [ {left: '\\\\(', right: '\\\\)', display: false}, {left: '$$', right: '$$', display: true} ] });">
  </script>
  <style>
    body {
      font-family: sans-serif;
      padding: 2em;
    }
  </style>
</head>
<body>
"""

latex_html_ends = r"""</body>
</html>
"""

def latex_to_html(latex_str_list, path):
  with open(path, "w") as file:
      file.write(latex_html_headers)
      file.write('\n')
      for s in latex_str_list:
        file.write(f'<p>$$ {s} $$</p>\n')
      file.write('\n')
      file.write(latex_html_ends)

if __name__ == '__main__':
  # Example: w * 2 + 3
  expr_tree = Ordinal(
    Node(
      '+',
      Node(
        '*',
        Node('w'),
        Node('2')
      ),
      Node('3'),
    )
  )
  print(expr_tree)  # Output: Infix Expression: (3 + (w * 2))
  latex_to_html([
    expr_tree.to_latex(),
    Ordinal(Node('1')).fundamental_sequence_display(3),
    Ordinal(Node('w')).fundamental_sequence_display(3),
    Ordinal.from_str('w^2+w').fundamental_sequence_display(3),
  ], './test.html')
