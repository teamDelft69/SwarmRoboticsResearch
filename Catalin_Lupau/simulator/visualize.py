from unittest import result

from flask import Flask, render_template, url_for

from controllers.visualization_controllers import (get_all_saved_simulations,
                                                   get_all_simulation_results,
                                                   get_saved_simulation,
                                                   get_simulation_result)

app = Flask(__name__)

@app.route('/simulations_and_results')
def get_simulations_and_results():
  return {'simulations': get_all_saved_simulations(), 'results': get_all_simulation_results()}


@app.route('/simulation/<simulation_name>')
def get_simulation(simulation_name):
  return get_saved_simulation(simulation_name)


@app.route('/result/<simulation_name>')
def get_result(simulation_name):
  return get_simulation_result(simulation_name)

@app.route('/vis/<simulation_name>')
def get_visualization(simulation_name):
  return render_template('simulation_visualizer.html', simulation_name=simulation_name)

@app.route('/')
def get_menu():
  return render_template('menu.html')
