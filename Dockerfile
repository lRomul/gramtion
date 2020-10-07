FROM ghcr.io/osai-ai/dokai:20.09-pytorch

WORKDIR /workdir

RUN pip3 install --no-cache-dir \
    git+https://github.com/ruotianluo/ImageCaptioning.pytorch.git@cd651fafa56e33a1d77ba1493c9785d766daa828 \
    gdown==3.12.2 \
    yacs==0.1.8 \
    requests==2.24.0 \
    ipywidgets==7.5.1

RUN git clone --depth 1 -b master --single-branch https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git &&\
    cd vqa-maskrcnn-benchmark &&\
    git checkout 4c168a637f45dc69efed384c00a7f916f57b25b8 &&\
    python3 setup.py build &&\
    python3 setup.py develop
ENV PYTHONPATH "${PYTHONPATH}:/workdir/vqa-maskrcnn-benchmark"

RUN mkdir model_data && cd model_data &&\
    gdown --id 1VmUzgu0qlmCMqM1ajoOZxOXP3hiC_qlL &&\
    gdown --id 1zQe00W02veVYq-hdq5WsPOS3OPkNdq79

RUN wget -O model_data/detectron_model.pth \
    https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth &&\
    wget -O model_data/detectron_model.yaml \
    https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml

COPY . /workdir
