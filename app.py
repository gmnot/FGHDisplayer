# app.py
from flask import Flask, request, render_template
from ordinal import Node, Ordinal, FGH

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  result = ''
  if request.method == 'POST':
    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')
    result = Ordinal.from_str(text1).fundamental_sequence_display(int(text2))
  return render_template('index.html', result=result)

if __name__ == '__main__':
  app.run(debug=True)
