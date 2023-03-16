cd docker
docker build . -t gain_pytorch_lightning
docker-compose up -d
container_id=$(docker ps -aqf "name=^docker-container_gain")
echo "Container ID: $container_id"
docker exec -it "$container_id" bash