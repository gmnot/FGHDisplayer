from __future__ import annotations
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

  def fundamental_sequence_at(self, n):
    if self.is_atomic():
      if self.is_natural():
        return self.value
      assert self.value == 'w'
      return str(n)
    assert 0

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

  def to_latex(self):
    return self.root.to_latex()

  def fundamental_sequence_at(self, n):
    assert self.root is not None
    return self.root.fundamental_sequence_at(n)
  
  def fundamental_sequence_display(self, n):
    fs = self.fundamental_sequence_at(n)
    return f'{self.to_latex()}[{n}]={fs}'

# Example: Construct an expression 3 + (w * 2)
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

latex_html_ends = r"""


</body>
</html>
"""

def latex_to_html(latex_str_list, path):
  with open(path, "w") as file:
      file.write(latex_html_headers)
      for s in latex_str_list:
        file.write(f'<p>$$ {s} $$</p>')
      file.write(latex_html_ends)

print(expr_tree)  # Output: Infix Expression: (3 + (w * 2))
latex_to_html([
  expr_tree.to_latex(),
  Ordinal(Node('1')).fundamental_sequence_display(3),
  Ordinal(Node('w')).fundamental_sequence_display(3),
], './test.html')
