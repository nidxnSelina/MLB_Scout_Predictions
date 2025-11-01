
FROM r-base:4.4.0

# make R non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# set workdir
WORKDIR /app

# system deps (for some R packages; keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# copy installer
COPY install_packages.R /app/install_packages.R
RUN Rscript /app/install_packages.R

# copy main R script
COPY main.R /app/main.R

# default command: run the R script
CMD ["Rscript", "/app/main.R"]

