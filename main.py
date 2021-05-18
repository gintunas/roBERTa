from flask import Flask, jsonify, request, render_template

app = Flask(__name__, template_folder="templates", static_folder='static')

from transformers import AutoModelForQuestionAnswering, pipeline, AutoTokenizer

model_checkpoint = "roberta-large"

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

nlp_qa = pipeline('question-answering', model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		context = request.form['context']
		question = request.form['question']

		answer = nlp_qa(context=context, question=question)
		return render_template('index.html', answer=answer)

@app.route('/predict', methods=['GET'])
def calculateGet():
	response = {
		"info": "Please send a POST request to this URL with context and question as JSON."
	}
	return jsonify(response)


if __name__ == '__main__':
	app.run()
