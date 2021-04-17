FROM ghcr.io/osai-ai/dokai:21.03-pytorch

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 7518
COPY . /workdir
CMD ["python", "src/twitter_bot.py"]
