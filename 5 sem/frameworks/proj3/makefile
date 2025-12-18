# Запуск контейнеров через docker-compose
run:
	docker compose up -d

build:
	docker compose build --no-cache

# Остановка и удаление контейнеров
stop:
	docker compose down

flush:
	docker compose down -v

# Вывести логи всех контейнеров
logs:
	docker compose logs -f

# Перезапуск
reload:
	docker compose down -v && docker compose up --build

# Ребилд без кэша
reboot:
	docker compose down -v && docker compose build --no-cache && docker compose up

# Пруннинг контейнеров
prune:
	docker compose down -v && docker volume prune && docker image prune && docker container prune && docker network prune

# Очистка всего (осторожно, можно удалить и несвязанные контейнеры)

absdelete:
	docker compose down -v && docker volume rm $(docker volume ls) && docker image rm $(docker image ls) && docker container rm $(dockere container ls)

netdelete:
	docker compose down -v && docker network rm $(docker network ls)

lazyd:
	cd lazydocker/ && go run main.go
