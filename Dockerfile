FROM rust:1.49.0
RUN apt update
RUN apt install -y cmake libclang-dev libc++-dev gcc-multilib
WORKDIR /app
