FROM allennlp/commit:d78daa44cc90b3ea9090fe6c24e788ebafe84496

WORKDIR /stage/allennlp

# Install the requirements
COPY requirements.txt .

RUN pip install -r requirements.txt -U

COPY contexteval/ contexteval/
COPY experiment_configs/ experiment_configs/
COPY scripts/ scripts/
COPY tests/ tests/

LABEL maintainer="nfliu@cs.washington.edu"

ENTRYPOINT []
CMD ["/bin/bash"]
