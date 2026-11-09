# rocker/r-ver pins R 4.4.1 and a dated CRAN snapshot, so package versions are reproducible
FROM rocker/r-ver:4.4.1

WORKDIR /app

COPY install_packages.R .
RUN Rscript install_packages.R

COPY run_model.R .

# The dataset is NOT copied into the image; it is mounted at runtime to /app/data
ENV DATA_DIR=/app/data

CMD ["Rscript", "run_model.R"]
