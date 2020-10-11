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
all: stop build run

.PHONY: build
build:
	docker build -t $(NAME) .

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

.PHONY: run
run:
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
