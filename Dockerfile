FROM rust:1.68.2
RUN apt update
RUN apt install -y cmake libclang-dev libc++-dev gcc-multilib
WORKDIR /app
