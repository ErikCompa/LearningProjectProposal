# frontend
FROM node:25.2.1-alpine3.21 AS frontend
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# backend
FROM python:3.14.2-slim-trixie AS backend
WORKDIR /code

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/

# final
COPY --from=frontend /web/dist ./app/static
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]