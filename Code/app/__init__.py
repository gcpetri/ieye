import os
from flask import Flask, render_template, send_from_directory
from app.Controllers.main import main_blueprint


def create_app():
    app = Flask(__name__)

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(app.root_path, 'static/favicon.ico', mimetype='image/vnd.microsoft.icon')

    @app.route('/test')
    def test_page():
        return render_template('test.html')

    @app.route("/result")
    def show_result_page():
        return render_template('result.html')

    @app.route("/login_result")
    def show_login_result_page():
        return render_template('login_result.html')

    @app.route("/")
    def hello_world():
        return render_template('index.html')
    
    app.register_blueprint(main_blueprint)

    return app