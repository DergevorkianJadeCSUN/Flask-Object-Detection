from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'

    from .views import views
    from .restapi import  restapi
    app.register_blueprint(views, url_prefix='/')
    restapi.register_blueprint(views, url_prefix='/')

    return app