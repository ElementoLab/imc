from rocker/r-base:3.6.3

# RUN wget https://github.com/qmiao19/CytoSpill/archive/master.zip

# RUN unzip master.zip

# WORKDIR CytoSpill-master

# RUN R CMD INSTALL .


RUN R --vanilla -e "install.packages('devtools', repo='http://cran.rstudio.com/')"

RUN R --vanilla -e "devtools::install_github('KChen-lab/CytoSpill')"

CMD ["R"]
