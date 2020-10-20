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
		-v $(shell pwd)/.env:/workdir/.env \
		--name=$(NAME) \
		$(NAME) \
		python src/twitter_bot.py

.PHONY: run-dev
run-dev:
	docker run --rm -dit \
		$(OPTIONS) \
		$(GPUS_OPTION) \
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
	docker exec -it $(NAME) $(COMMAND)
