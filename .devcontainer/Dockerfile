FROM python:3.11-slim-bookworm

# System libraries required by geospatial Python packages
# (GDAL → fiona/geopandas, GEOS → shapely, PROJ → pyproj, spatialindex → rtree)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
