FROM node:24-bookworm-slim

WORKDIR /repo
ENV NODE_ENV=development

COPY . .

RUN npm ci --include=dev
RUN npm run build --workspace=api

ENV NODE_ENV=production
ENV PORT=10000

EXPOSE 10000

CMD ["sh", "-c", "if [ -f apps/api/dist/main.js ]; then exec node apps/api/dist/main.js; elif [ -f apps/api/dist/src/main.js ]; then exec node apps/api/dist/src/main.js; else echo 'NestJS main.js not found'; exit 1; fi"]
