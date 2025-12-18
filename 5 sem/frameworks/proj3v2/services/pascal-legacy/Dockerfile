FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    fpc \
    fpc-source \
    libncurses-dev \
    libc6-dev \
    postgresql-client \
    ca-certificates \
    zip \
    unzip \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Установка openpyxl для генерации XLSX
# Используем --break-system-packages для обхода PEP 668 в Debian 12
RUN pip3 install --no-cache-dir --break-system-packages openpyxl

WORKDIR /opt/legacy

COPY legacy.pas ./
COPY csv_to_xlsx.py ./
COPY run.sh /usr/local/bin/run_legacy.sh
RUN chmod +x /usr/local/bin/run_legacy.sh
RUN chmod +x csv_to_xlsx.py

CMD ["/usr/local/bin/run_legacy.sh"]
