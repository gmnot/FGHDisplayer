
from enum import Enum

contact_request = "please report your inputs (and environments) to fghdisplayer@gmail.com"

def latex_to_block(s):
  return f' $$ {s} $$ '

class OutType(Enum):
  DIV    = 1
  PLAIN  = 2
