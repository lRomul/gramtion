NAME?=gramtion
COMMAND?=bash
OPTIONS?=

GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all
all: stop build run logs

.PHONY: build
build:
	docker build -t $(NAME) .

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

.PHONY: run
run:
	docker run -d --restart=always \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		--env-file .env \
		-v $(shell pwd)/google_key.json:/workdir/google_key.json \
		--name=$(NAME) \
		$(NAME)

.PHONY: run-dev
run-dev:
	docker run --rm -dit \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		--env-file .env \
		-v $(shell pwd)/google_key.json:/workdir/google_key.json \
		--name=$(NAME) \
		$(NAME) \
		$(COMMAND)
	docker attach $(NAME)

.PHONY: attach
attach:
	docker attach $(NAME)

.PHONY: logs
logs:
	docker logs -f $(NAME)

.PHONY: exec
exec:
	docker exec -it $(OPTIONS) $(NAME) $(COMMAND)
