import logging
import os


from dotenv import load_dotenv
from quart import Quart
from quart_cors import cors


def create_app():
    # We do this here in addition to gunicorn.conf.py, since we don't always use gunicorn
    load_dotenv(override=True)


    # Set Logging Level based on environment
    if os.getenv("RUNNING_IN_PRODUCTION"):
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)


    app = Quart(__name__)
    # app = cors(app, allow_origin="https://icy-island-0d9782c0f.1.azurestaticapps.net", allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    # app = cors(app, allow_origin="https://icy-island-0d9782c0f.1.azurestaticapps.net", allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], allow_headers=["*"], allow_credentials=True)
    app = cors(app, allow_origin="*", allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], allow_headers=["*"])

    from . import chat  # noqa

    # YUBI: new code adds api prefix so all API routes are now under /api/  (e.g., /api/process_pdf, /api/get_result_by_timestamp)
    # app.register_blueprint(chat.bp)
    app.register_blueprint(chat.bp, url_prefix="/api")


    return app