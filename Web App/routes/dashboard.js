import { Router } from "express";
import * as prefRoute from "../data/users/setPreference.js";
import routeError from "./routeerror.js";
import multer from 'multer';
import axios from 'axios';
import uploadEEGData from "../data/users/uploadEEGData.js"

const router = Router();
const upload = multer();

router
  .get("/", async (req, res) => {
    try {
      if (req.session.type === "Admin") {
        return res.render("admin/dashboard");
      }

      let renderObjs = {
        showoutput: false,
        eegdata: [],
      };
      return res.render("public/dashboard", renderObjs);

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
  })
  .post('/upload', upload.single('file'), async (req, res) => {
    try {
      // Create a Blob object from the Uint8Array buffer
      const fileBlob = new Blob([req.file.buffer], { type: req.file.mimetype });

      // Create FormData object
      const formData = new FormData();
      formData.append('file', fileBlob, req.file.originalname);

      // Forward the file to Flask application
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      let uploadData = await uploadEEGData(response.data.result, req);

      return res.json(response.data.result)

    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

export default router;
