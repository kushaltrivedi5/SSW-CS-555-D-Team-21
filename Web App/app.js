import "dotenv/config";
import express from "express";
import session from "express-session";
import MongoStore from "connect-mongo";
import exphbs from "express-handlebars";
import { fileURLToPath } from "url";
import { dirname } from "path";
import route from "./routes/index.js";
import SMTPConnect from "./config/smptConnection.js";
import { dbConnection } from "./config/mongoConnection.js";

const smptconnection = SMTPConnect();
const databaseconnection = dbConnection();


const __filename = fileURLToPath(import.meta.url);
export const __dirname = dirname(__filename);

const app = express();

const iconsDir = express.static(__dirname + "/static/icons");
app.use("/icons", iconsDir);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use(
  session({
    secret: process.env.CookieSecret,
    resave: false,
    saveUninitialized: false,
    store: MongoStore.create({
      mongoUrl: process.env.mongoServerUrl,
      dbName: process.env.mongoDbname,
    }),
  })
);

app.use("/", (req, res, next) => {
  if (req.body._csrf) {
    req.headers["x-csrf-token"] = req.body._csrf;
    delete req.body._csrf;
  }
  next();
});

const eqHelper = function (a, b) {
  return a === b;
};

const gtHelper = function (a, b) {
  return a > b;
};

const gtD = function (a, b) {
  const x = new Date(a);
  const y = new Date(b);
  return x > y;
};

const ifUserType = function (roleString, session_type, options) {
  const roleArray = roleString.split(',');

  if (Array.isArray(roleArray) && roleArray.includes(session_type)) {
    return options.fn(this);
  } else {
    return options.inverse(this);
  }
}

const handlebars = exphbs.create({
  defaultLayout: "main",
  partialsDir: ["views/partials/"],
  helpers: { eq: eqHelper, gt: gtHelper, gtD: gtD, ifUserType: ifUserType },
});

app.engine("handlebars", handlebars.engine);
app.set("view engine", "handlebars");
app.set("views", "./views");


// Load routes
route(app);

smptconnection.verify(function (error, success) {
  if (error) {
    console.log(error);
    throw "Failed to connect to SMTP";
  } else {
    console.log("Connected to SMTP Server");
  }
});

if (await databaseconnection) {
  console.log("Connected to Database Server");
} else {
  throw "Failed to connect to database";
}

app.listen(8080, () => {
  console.log("Running web server on port 8080");
  console.log(`http://localhost:8080/`);
});
