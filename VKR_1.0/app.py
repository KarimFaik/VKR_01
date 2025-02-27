from flask import Flask, request, jsonify, render_template
from model import question_answer

app = Flask(__name__)

# Загрузка текста из файла
def load_text():
    with open("text.txt", "r", encoding="utf-8") as file:
        return file.read()

# Маршрут для главной страницы
@app.route("/")
def home():
    return render_template("index.html")

# Маршрут для обработки вопроса
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Вопрос не предоставлен"}), 400

    # Загрузка текста из файла
    text = load_text()
    if not text:
        return jsonify({"error": "Текст не найден"}), 400

    # Поиск ответа
    answer = question_answer(text, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)