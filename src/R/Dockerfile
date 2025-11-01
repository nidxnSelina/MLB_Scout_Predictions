FROM r-base:4.4.0

ENV DEBIAN_FRONTEND=noninteractive

# minimal system deps some R packages commonly need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install R packages
COPY src/R/install_packages.R ./install_packages.R
RUN Rscript --vanilla install_packages.R

# copy the runner
COPY src/R/run.R ./run.R

# if run.R expects ./data, mount it at /app/data when running
CMD ["Rscript", "--vanilla", "run.R"]
