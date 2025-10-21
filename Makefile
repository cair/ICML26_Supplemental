.devcontainer/.env:
	@echo "UID=$(id -u)"   > $@
	@echo "GID=$(id -g)"  >> $@
	@echo "UNAME=$(id -un)" >> $@

up: .devcontainer/.env
	docker compose -f .devcontainer/docker-compose.yml build dev

