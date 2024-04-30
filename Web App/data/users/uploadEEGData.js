import { users } from "../../config/mongoCollections.js";
import { ObjectId } from "mongodb";

async function uploadEEGData(response, req) {
    try {
        if (req.session.userid) {
            const usercol = await users();

            // Generate a new ObjectId for each entry
            const objectId = new ObjectId();

            const myJson = {
                _id: objectId, // Add the new ObjectId to the JSON object
                filename: req.file.originalname,
                data: response
            }

            const existingUser = await usercol.findOne({ _id: new ObjectId(req.session.userid) });
            let eegdataArray = existingUser.eegdata || []; // Initialize as an empty array if 'eegdata' does not exist or is not an array

            eegdataArray.push(myJson);

            const result = await usercol.updateOne(
                { _id: new ObjectId(req.session.userid) },
                { $set: { eegdata: eegdataArray } }
            );

            if (!result) {
                throw { status: 404, message: "No such user" };
            }

            return result;
        }
    }
    catch (e) {
        console.log(e);
    }
}

export default uploadEEGData;
