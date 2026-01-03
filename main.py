from flask import Flask, render_template, request, jsonify
import sys
import os

# 1. Definición de rutas base (Solución al NameError)
base_dir = os.path.abspath(os.path.dirname(__file__))

# 2. Configuración de Flask (Una sola vez, al principio)
app = Flask(__name__, 
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))

# 3. Corrección de Ruta para la carpeta 'python'
sys.path.append(os.path.join(base_dir, 'python'))

# 4. Importaciones de tus módulos de distribución
from Distribucion_discretas import DistributionFactory as DiscreteFactory
from Distribucion_continuas import ContinuousDistributionFactory as ContinuousFactory

# --- RUTAS DE NAVEGACIÓN ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/discretas')
def discretas():
    return render_template('discretas.html')

@app.route('/continuas')
def continuas():
    return render_template('continuas.html')

# --- LÓGICA DE DISTRIBUCIONES ---

def get_distribution_instance(dist_name, params):
    name = dist_name.lower()
    
    if name in ["gamma", "exponential"]:
        return ContinuousFactory.create(name, **params)
    
    if name == "normal":
        # Selección inteligente: si usa 'mu' es la discreta, si no, la continua
        if "mu" in params:
            return DiscreteFactory.create("normal", **params)
        else:
            return ContinuousFactory.create("normal", **params)
            
    return DiscreteFactory.create(name, **params)

# --- ENDPOINTS API ---

@app.route('/probability', methods=['POST'])
def probability():
    data = request.get_json()
    try:
        dist_name = data.get('distribution')
        value = data.get('value')
        
        # Filtramos parámetros para la instancia
        params = {k: v for k, v in data.items() if k not in ['distribution', 'value', 'acc']}
        
        dist = get_distribution_instance(dist_name, params)
        result = dist.getProbability(value, acc=data.get('acc', False))
        
        return jsonify({"probability": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/sample', methods=['POST'])
def sample():
    data = request.get_json()
    try:
        dist_name = data.get('distribution')
        n = data.get('cardinality', 1)
        
        params = {k: v for k, v in data.items() if k not in ['distribution', 'cardinality']}
        
        dist = get_distribution_instance(dist_name, params)
        samples, density_values = dist.getSample(n)
        
        return jsonify({
            "sample": samples,
            "pmf_pdf_values": density_values
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)