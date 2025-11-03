#!/bin/bash

set -e

cd "$(dirname "$0")"

docker-compose -f .docker/docker-compose.yaml up -d
docker exec -it drake bash
