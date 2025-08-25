# YUBI: I created this file to be the entry point for the Quart app
# It imports the create_app function from the __init__.py file and calls it to create
# This enables gunicorn to run the Quart app with the factory pattern

from . import create_app
app = create_app()