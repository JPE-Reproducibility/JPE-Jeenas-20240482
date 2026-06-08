# Script that installs the R packages required

install.packages("remotes", repos = "https://cloud.r-project.org")
remotes::install_version("plm", version = "2.4-1", repos = "https://cloud.r-project.org")
remotes::install_version("plyr", version = "1.8.6", repos = "https://cloud.r-project.org")
remotes::install_version("lmtest", version = "0.9-38", repos = "https://cloud.r-project.org")
remotes::install_version("stargazer", version = "5.2.2", repos = "https://cloud.r-project.org")
remotes::install_version("lfe", version = "2.8-6", repos = "https://cloud.r-project.org")
remotes::install_version("moments", version = "0.14", repos = "https://cloud.r-project.org")
remotes::install_version("xtable", version = "1.8.4", repos = "https://cloud.r-project.org")
remotes::install_version("fixest", version = "0.10.1", repos = "https://cloud.r-project.org")