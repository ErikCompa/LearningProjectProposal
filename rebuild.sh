#!/bin/bash

echo "Cleaning old builds..."
rm -rf app/static
rm -rf web/dist

echo "Building frontend..."
cd web
npm run build

echo "Copying dist to app/static..."
cp -r dist ../app/static

echo "Starting server..."
cd ..
uvicorn app.main:app --reload