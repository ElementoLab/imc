FROM cellprofiler/cellprofiler

RUN pip install tifffile==0.9.2

RUN git clone https://github.com/BodenmillerGroup/ImcPluginsCP && \
	cd ImcPluginsCP && \
	git checkout develop-cp3 && \
	cp plugins/*.py /usr/local/src/CellProfiler/plugins/ && \
	cp -r plugins /root/./plugins
RUN git clone https://github.com/BodenmillerGroup/ImcSegmentationPipeline

ENTRYPOINT ["/init", "cellprofiler"]
