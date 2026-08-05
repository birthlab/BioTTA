FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG APP_UID=1000
ARG APP_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HOME=/home/biotta \
    MPLCONFIGDIR=/home/biotta/.cache/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" biotta \
    && useradd --create-home --uid "${APP_UID}" --gid "${APP_GID}" --shell /bin/bash biotta

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=biotta:${APP_GID} . /app

RUN mkdir -p /data/input /data/output "${MPLCONFIGDIR}" \
    && chown -R biotta:${APP_GID} /data /home/biotta /app

USER biotta

CMD ["python", "main.py", "--help"]
