# Build stage
FROM rust:1.96-slim AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home memrust \
    && mkdir /data \
    && chown memrust:memrust /data
COPY --from=build /app/target/release/memrust /usr/local/bin/memrust
USER memrust
VOLUME /data
EXPOSE 7700
# /healthz is unauthenticated on purpose — /health needs a key once
# --api-key is set, and a probe should not hold credentials.
HEALTHCHECK --interval=30s --timeout=3s CMD curl -sf http://127.0.0.1:7700/healthz || exit 1
ENTRYPOINT ["memrust"]
CMD ["serve", "--addr", "0.0.0.0:7700", "--data-dir", "/data"]
