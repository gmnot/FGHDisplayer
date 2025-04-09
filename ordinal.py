class Node:
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


class Ordinal:
  def __init__(self, root=None):
    """
    Expression represents the entire mathematical expression as a binary tree.
    :param root: The root node of the expression (main operator)
    """
    self.root = root

  @classmethod
  def ord_to_latex(cls, ord):
    """
    basic ord symbol to latex.
    such as 1,2,3..., w, \varepsilon
    """
    if '0' <= ord <= '9':
      return ord
    assert ord == 'w'
    return r'\omega'
  
  @classmethod
  def op_to_latex(cls, op):
    match op:
      case '+':
        return '+'
      case '*':
        return r'\cdot'
      case '^':
        return '^'
      case _:
        assert 0, op

  def __str__(self):
    """
    Converts the expression tree to an infix string representation (with parentheses).
    """
    return self._to_infix(self.root)

  def _to_infix(self, node):
    """
    Recursively converts the expression tree to an infix string.
    :param node: The current node in the tree
    """
    if node is None:
      return ""
    # If the node is a leaf (a natural number or 'w')
    if node.left is None and node.right is None:
      return str(node.value)
    # If the node is an operator
    left_expr  = self._to_infix(node.left)
    right_expr = self._to_infix(node.right)
    return f"({left_expr} {node.value} {right_expr})"

  def _to_latex(self, node):
    if node is None:
      return ""
    # If the node is a leaf (a natural number or 'w')
    if node.left is None and node.right is None:
      return self.ord_to_latex(node.value)
    # If the node is an operator
    left_expr  = self._to_latex(node.left)
    right_expr = self._to_latex(node.right)
    return '{' + left_expr + '}' + self.op_to_latex(node.value) + '{' + right_expr + '}'

  def to_latex(self):
    return '$$ ' + self._to_latex(self.root) + ' $$'

# Example: Construct an expression 3 + (w * 2)
expr_tree = Ordinal(
  Node(
    '+',
    Node('3'),
    Node(
      '*',
      Node('w'),
      Node('2')
    )
  )
)

print(expr_tree)  # Output: Infix Expression: (3 + (w * 2))
print(expr_tree.to_latex())
