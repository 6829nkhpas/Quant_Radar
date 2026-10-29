# Quant_Radar image: built in CI and pushed to GHCR, so the box only pulls it.
#
# Two stages. The runtime is distroless/cc (glibc + libgcc, no shell, no package
# manager): the binary needs nothing else, because TLS is rustls with bundled
# webpki roots, so not even the system CA store is read.

FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY quant-core ./quant-core
# Dependencies in their own layer: build once against a stub main, so a source-only
# change does not recompile tokio/axum/reqwest.
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release --locked && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release --locked

FROM gcr.io/distroless/cc-debian12:nonroot
LABEL org.opencontainers.image.source="https://github.com/6829nkhpas/Quant_Radar"
COPY --from=build /src/target/release/quant_radar /quant_radar
EXPOSE 8090
ENTRYPOINT ["/quant_radar"]
