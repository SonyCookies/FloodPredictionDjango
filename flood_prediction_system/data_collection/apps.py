import os
from django.apps import AppConfig

class YourAppConfig(AppConfig):
    name = 'data_collection'

    def ready(self):
        if os.environ.get('RUN_MAIN', None) != 'true':  # Only run once, avoid Django's auto-reloader
            from .scheduler import start
            start()
