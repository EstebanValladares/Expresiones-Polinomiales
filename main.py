from flask import Flask, request, jsonify
# Importamos las fábricas desde la carpeta 'python'
# Asegúrate de tener un archivo vacío llamado __init__.py en esa carpeta
from python.Distribucion_discretas import DistributionFactory as DiscreteFactory
from python.Distribucion_continuas import ContinuousDistributionFactory as ContinuousFactory

app = Flask(__name__)

def get_factory(category):
    """Retorna la fábrica correspondiente según la categoría."""
    if category.lower() == 'discreta':
        return DiscreteFactory
    elif category.lower() == 'continua':
        return ContinuousFactory
    else:
        raise ValueError(f"Categoría '{category}' no válida. Use 'discreta' o 'continua'.")

@app.route('/get_probability', methods=['POST'])
def get_probability():
    data = request.get_json()
    
    dist_type = data.get('type')      # Ej: 'normal', 'poisson', 'gamma'
    category = data.get('category')    # 'discreta' o 'continua'
    value = data.get('value')         # El punto X a evaluar
    acc = data.get('acc', False)      # Booleano para acumulada
    params = data.get('params', {})   # Diccionario de parámetros (mu, sd, k, etc.)

    try:
        factory = get_factory(category)
        dist = factory.create(dist_type, **params)
        
        result = dist.getProbability(value, acc=acc)
        return jsonify({
            "status": "success",
            "type": dist_type,
            "category": category,
            "probability": result
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/get_sample', methods=['POST'])
def get_sample():
    data = request.get_json()
    
    dist_type = data.get('type')
    category = data.get('category')
    n = data.get('n', 1)              # Cantidad de muestras
    params = data.get('params', {})

    try:
        factory = get_factory(category)
        dist = factory.create(dist_type, **params)
        
        samples, pdf_values = dist.getSample(n)
        return jsonify({
            "status": "success",
            "type": dist_type,
            "samples": samples,
            "density_or_pmf": pdf_values
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # El servidor corre en http://127.0.0.1:5000
    app.run(debug=True, port=5000)