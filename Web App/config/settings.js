import "dotenv/config";

export const mongoConfig = {
  serverUrl: process.env.mongoServerUrl,
  database: process.env.mongoDbname,
};
