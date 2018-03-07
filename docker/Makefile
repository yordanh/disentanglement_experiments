help:
	@cat Makefile

DATA?="${HOME}/shared_data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)

build:
	docker build -t keras --build-arg python_version=2.7.13 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it --rm -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras bash

ipython: build
	$(DOCKER) run -it --rm -v $(SRC):/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	$(DOCKER) run -it --rm -v $(SRC):/src -v $(DATA):/data --net=host --env KERAS_BACKEND=$(BACKEND) keras

test: build
	$(DOCKER) run -it --rm -v $(SRC):/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)

