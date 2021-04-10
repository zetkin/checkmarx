FROM node:slim
WORKDIR /usr/src/app
COPY package.json package-lock.json ./
RUN npm update
RUN npm install
COPY . .
