# ------------------- Stage 0: Base Stage ------------------------------
FROM python:3.11-alpine AS base

WORKDIR /code

# Install tini, a tiny init for containers
RUN apk add --update --no-cache tini

# Install required packages for cryptography package
# https://cryptography.io/en/latest/installation/#building-cryptography-on-linux
RUN apk add gcc musl-dev python3-dev libffi-dev openssl-dev cargo pkgconfig tesseract-ocr tesseract-ocr-data-eng

# ------------------- Stage 1: Build Stage ------------------------------
FROM base AS build

# Build argument to force cache bust
ARG BUILD_ID
ENV BUILD_ID=${BUILD_ID}

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .
# ------------------- Stage 2: Final Stage ------------------------------
FROM base AS final

RUN addgroup -S app && adduser -S app -G app

COPY --from=build --chown=app:app /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=build --chown=app:app /usr/local/bin /usr/local/bin
COPY --from=build --chown=app:app /code /code

USER app

EXPOSE 50505

ENTRYPOINT ["tini", "gunicorn", "quartapp:create_app()"]
# using this file will hardcode the port to 50505
# ENTRYPOINT ["tini", "gunicorn", "-c", "gunicorn.conf.py", "quartapp:create_app()"]
# YUBI: use factory flag because create_app is a function not an attribute, double check this bc it might be for flask apps but not quart
# ENTRYPOINT ["tini", "gunicorn", "-c", "gunicorn.conf.py", "quartapp:create_app", "--factory"]
# YUBI: use wsgi file to run the app, this is the entry point for the Quart app
# ENTRYPOINT ["tini", "gunicorn", "-c", "gunicorn.conf.py", "quartapp.wsgi:app"]