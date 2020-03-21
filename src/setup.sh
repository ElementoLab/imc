

docker pull cellprofiler/cellprofiler
xhost +local:root
docker build -t afrendeiro:cellprofiler . 

# docker run \
#     -v /home/afr/projects/hyperion-cytof:/hyperion-cytof:rw \
#     -v /home/afr/Documents/workspace/clones/ImcSegmentationPipeline/:/ImcSegmentationPipeline:ro \
#     -v /home/afr/Documents/workspace/clones/ImcPluginsCP/:/ImcPluginsCP:ro \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
#     afrendeiro:cellprofiler
#     # cellprofiler/cellprofiler:latest ""


# docker run \
#     -v /home/afr/projects/hyperion-cytof:/hyperion-cytof:rw \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     1c105475d775 \
#     "--plugins-directory /usr/local/src/CellProfiler/ImcPluginsCP/plugins \
#     -c -r -p /usr/local/src/CellProfiler/ImcSegmentationPipeline/cp3_pipelines/1_prepare_ilastik.cppipe"
#     # -v /home/afr/Documents/workspace/clones/ImcSegmentationPipeline/:/ImcSegmentationPipeline:rw \
#     # -v /home/afr/Documents/workspace/clones/ImcPluginsCP/:/ImcPluginsCP:rw \
#     # cellprofiler/cellprofiler:latest ""


docker run \
    -v /home/afr/projects/hyperion-cytof:/hyperion-cytof:rw \
    -v /home/afr/Documents/workspace/clones/ImcSegmentationPipeline/:/ImcSegmentationPipeline:ro \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    afrendeiro:cellprofiler ""


# docker run \
#     -v /home/afr/projects/hyperion-cytof:/hyperion-cytof:rw \
#     -v /home/afr/Documents/workspace/clones/ImcSegmentationPipeline/:/ImcSegmentationPipeline:ro \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
#     cellprofiler:centos7 ""


singularity run docker://cellprofiler/cellprofiler ""

# /root/./plugins

# ^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])

# .*\.(?P<roi>.*?)\.(?P<marker>.*?).ome.tiff


PRE_TRAINED_MODEL=fluidigm_example_data/fluidigm_example_data.ilp
python src/pipeline.py \
    -m $PRE_TRAINED_MODEL \
    --csv-pannel data/case_b_mcd/panel_markers.panelB.csv \
    -i data/case_b_mcd \
    -s 1 \
    processed/case_b
