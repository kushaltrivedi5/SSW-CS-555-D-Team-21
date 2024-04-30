import { users } from "../../config/mongoCollections.js";
import { ObjectId } from "mongodb";

async function fetchEEGData(req) {
    try {
        if (req.session.userid) {
            const usercol = await users();

            const existingUser = await usercol.findOne({ _id: new ObjectId(req.session.userid) });
            let eegdataArray = existingUser.eegdata || []; // Initialize as an empty array if 'eegdata' does not exist or is not an array


            return eegdataArray;
        }
    }
    catch (e) {
        console.log(e);
    }
}

async function fetchEEGDataInfo(req) {
    try {
        if (req.session.userid) {
            const usercol = await users();

            const eegid = req.params.eegid; // Fetch eegid from request parameters

            const existingUser = await usercol.findOne({
                _id: new ObjectId(req.session.userid),
                "eegdata._id": new ObjectId(eegid) // Search for the eegid within the eegdata array
            });

            if (existingUser) {
                // Find the eegdata object with the given eegid
                const eegdataObject = existingUser.eegdata.find(eegdata => eegdata._id.toString() === eegid);
                return eegdataObject;
            } else {
                throw { status: 404, message: "User not found" };
            }
        }
    }
    catch (e) {
        console.log(e);
    }
}

export { fetchEEGData, fetchEEGDataInfo };
