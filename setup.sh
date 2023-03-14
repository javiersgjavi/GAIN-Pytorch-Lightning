docker build . -t gain_pytorch_lightning
docker-compose up -d
container_id=$(docker ps -aqf "name=^gain-pytorch-container")
echo "Container ID: $container_id"
docker exec -it "$container_id" bash