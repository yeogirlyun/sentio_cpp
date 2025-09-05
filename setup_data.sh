#!/bin/bash

# Sentio C++ Data Setup Script
# This script helps set up the data infrastructure and download market data

echo "=== Sentio C++ Data Setup ==="

# Check if config.env exists
if [ ! -f "config.env" ]; then
    echo "Creating config.env from template..."
    cp config.env.example config.env
    echo "Please edit config.env and add your Polygon.io API key"
    echo "You can get a free API key from: https://polygon.io/"
    exit 1
fi

# Load environment variables
source config.env

# Check if API key is set
if [ "$POLYGON_API_KEY" = "your_polygon_api_key_here" ] || [ -z "$POLYGON_API_KEY" ]; then
    echo "Please set your POLYGON_API_KEY in config.env"
    exit 1
fi

# Export the API key for poly_fetch
export POLYGON_API_KEY

echo "Using Polygon API key: ${POLYGON_API_KEY:0:8}..."

# Create data directories
mkdir -p data/equities data/audit

# Download QQQ family data for the last 30 days
echo "Downloading QQQ family data..."
./build/poly_fetch qqq $(date -d '30 days ago' '+%Y-%m-%d') $(date '+%Y-%m-%d') data/equities --rth

# Download some additional symbols
echo "Downloading additional symbols..."
./build/poly_fetch custom $(date -d '30 days ago' '+%Y-%m-%d') $(date '+%Y-%m-%d') data/equities --symbols "SPY,QQQ,IWM,GLD,TLT" --rth

echo "Data setup complete!"
echo "Available data files:"
ls -la data/equities/
