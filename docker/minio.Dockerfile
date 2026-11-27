FROM golang:1.24.9 AS builder
RUN CGO_ENABLED=0 GOMAXPROCS=2 go install -p 2 github.com/minio/minio@RELEASE.2025-10-15T17-29-55Z \
    && go clean -cache -modcache

FROM python:3.12.12-slim-bookworm
COPY --from=builder /go/bin/minio /usr/local/bin/minio
RUN useradd --create-home minio && mkdir /data && chown minio:minio /data
USER minio
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
