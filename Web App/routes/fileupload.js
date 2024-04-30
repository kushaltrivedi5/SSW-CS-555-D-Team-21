import { Router } from "express";
import * as prefRoute from "../data/users/setPreference.js";
import routeError from "./routeerror.js";
import { fetchEEGData, fetchEEGDataInfo } from "../data/users/fetchEEGData.js";

const router = Router();

router
  .get("/", async (req, res) => {
    try {
      if (req.session.type === "Admin") {
        return res.render("admin/dashboard");
      }

      const result = await fetchEEGData(req);
      let renderObjs = {
        eegdata: result
      };
      return res.render("public/fileupload", renderObjs);

    } catch (e) {
      routeError(res, e);
    }
  })
  .get("/:eegid", async (req, res) => {
    try {


      const result = await fetchEEGDataInfo(req);

      const occurrences = countOccurrences(result.data);
      let renderObjs = {
        eegdata: result,
        occurrences: occurrences
      };
      return res.render("public/vieweeg", renderObjs);

    } catch (e) {
      routeError(res, e);
    }
  })
  .post("/", async (req, res) => {
    try {
      const result = await prefRoute.setTheme(
        req.session.userid,
        req.body.darkmode
      );
      req.session.darkmode = req.body.darkmode;

      res.status(200).json({ message: "Theme updated successfully", result });
    } catch (e) {
      return res.status(e.status).json(e.message);
    }
  });

const countOccurrences = (textArray) => {
  const counts = {};

  textArray.forEach(text => {
    counts[text] = (counts[text] || 0) + 1;
  });

  return counts;
};
export default router;
