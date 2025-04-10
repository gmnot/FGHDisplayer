# app.py
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  result = ''
  if request.method == 'POST':
    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')
    result = text1 + text2  # 可以自定义拼接逻辑
  return render_template('index.html', result=result)

if __name__ == '__main__':
  app.run(debug=True)
