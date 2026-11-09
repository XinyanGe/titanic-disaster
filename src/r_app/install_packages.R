# Install pinned package versions
install.packages("remotes", repos = "https://cloud.r-project.org")

pkgs <- list(
  dplyr = "1.1.4",
  readr = "2.1.5"
)

for (p in names(pkgs)) {
  remotes::install_version(p, version = pkgs[[p]],
                           repos = "https://cloud.r-project.org",
                           upgrade = "never", quiet = TRUE)
}

# Fail the Docker build if anything did not install
for (p in names(pkgs)) {
  if (!requireNamespace(p, quietly = TRUE)) stop(paste("Failed to install", p))
  cat(sprintf("Installed %s %s\n", p, as.character(packageVersion(p))))
}
