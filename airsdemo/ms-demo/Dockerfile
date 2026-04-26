FROM python:3.12-slim


#ADD https://docker.art.code.pan.run:443/artifactory/generic-ep/certs/PANWEnterpriseCA2-G2-bundle.pem /usr/local/share/ca-certificates/PANWEnterpriseCA2-G2-bundle.pem
COPY cert.crt /usr/local/share/ca-certificates/cert.crt
RUN update-ca-certificates
RUN pip install uv

WORKDIR /app

