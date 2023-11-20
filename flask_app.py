from flask import Flask, jsonify, request
from prediction_function import predictStockMarket

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()

        # Access data using keys
        date = data.get('date')
        polarity = data.get('polarity')

        # Assuming calculate_value returns a float
        result = predictStockMarket(date, polarity)

        # Return the result as JSON
        return jsonify({'result': result})
    else:
        return jsonify({'message': 'Invalid request method'})

if __name__ == '__main__':
    app.run(debug=True)
