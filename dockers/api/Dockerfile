FROM python:3.9.7

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc-dev \
    unixodbc \
    libpq-dev

RUN apt-get install -y curl apt-transport-https

# Retrieves packages from Microsoft
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update

# Installs SQL drivers and tools
ENV ACCEPT_EULA=Y
RUN apt-get install -y msodbcsql17

# fix ssl connection issue
RUN apt-get install -y --no-install-recommends openssl
RUN sed -i 's,^\(MinProtocol[ ]*=\).*,\1'TLSv1.0',g' /etc/ssl/openssl.cnf
RUN sed -i 's,^\(CipherString[ ]*=\).*,\1'DEFAULT@SECLEVEL=1',g' /etc/ssl/openssl.cnf

RUN useradd -ms /bin/bash myuser
USER myuser

ENV PATH="/home/myuser/.local/bin:${PATH}"

COPY ./requirements.txt /requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --user -r /requirements.txt

COPY --chown=myuser:myuser . .