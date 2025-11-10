#!/bin/bash

docker-compose -f docker/docker-compose.yaml up -d
docker exec -it drake bash
