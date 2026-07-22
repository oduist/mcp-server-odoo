# Agent Instructions

## Docker Publishing

When rebuilding and publishing the Docker image for this repository, read the
image version from `pyproject.toml` `[project].version`. Publish both `latest`
and the exact version tag to Docker Hub as `oduist/odoo-mcp-server`.

Use a TOML parser rather than hard-coding the version:

```sh
VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
docker buildx build \
  --builder multiarch-builder \
  --platform linux/amd64,linux/arm64 \
  -t oduist/odoo-mcp-server:latest \
  -t "oduist/odoo-mcp-server:${VERSION}" \
  --push .
```

After publishing, verify the remote tags:

```sh
docker buildx imagetools inspect oduist/odoo-mcp-server:latest
docker buildx imagetools inspect "oduist/odoo-mcp-server:${VERSION}"
```
