tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
app = Flask("Text Summariser")

@app.route('/summarizer', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        min_length = int(request.form['min_length'])
        max_length = int(request.form['max_length'])
        summary = generate_summary(text, min_length, max_length)
        return render_template('index.html', summary=summary)
    return render_template('index.html')

def generate_summary(text, min_length, max_length):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, min_length=min_length, 
                             length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary 
