# retrieva

## Development setup
- Install `uv`
- Install `docker`
- To enable gpu support, follow these instuctions: https://docs.ollama.com/docker (install `nvidia-container-toolkit`)
- copy `.env.example` to `.env`
- run `uv sync`
- run `source .venv/bin/activate`
- run `docker compose -f infrastructure/docker-compose.yml up -d`

## Dockerized build
To just build and run the application in docker, run `docker compose up -d`

