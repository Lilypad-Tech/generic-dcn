FROM node:20-alpine AS build-env

RUN echo "installing apk packages" && \
  apk update && \
  apk upgrade && \
  apk add \
    bash \
    git \
    curl \
    openssh

WORKDIR /app

# root config
COPY ./*.json /app/
COPY ./yarn.lock /app/yarn.lock

# copy code
COPY . /app

# install modules
RUN yarn install
RUN yarn build

# frontend deploy
FROM nginx:alpine as deploy-frontend
LABEL maintainer="wings@lilypad.tech"
COPY ./nginx.conf /etc/nginx/nginx.conf
COPY --from=build-env /app/dist /www
