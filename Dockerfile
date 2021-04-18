FROM ghcr.io/osai-ai/dokai:21.03-pytorch

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN cd / && git clone --depth 1 --single-branch -b v1.1 https://www.github.com/kdexd/virtex &&\
    cd virtex && python3 setup.py develop

RUN python -c "from virtex import model_zoo;model = \
model_zoo.get('width_ablations/bicaptioning_R_50_L1_H2048.yaml', pretrained=True)"

EXPOSE 7518
COPY . /workdir
CMD ["python", "src/twitter_bot.py"]
