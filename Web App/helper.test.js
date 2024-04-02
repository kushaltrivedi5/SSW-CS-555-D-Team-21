// Import the modules you need to test
import * as helper from './helper.js';
import * as tester from "./config/mongoCollections.js";

// Define the test function
const test = (description, testFunction) => {
    console.log(`Running test: ${description}`);
    try {
        testFunction();
        console.log('Test passed.');
    } catch (error) {
        console.error('Test failed:', error.message);
    }
};

// Define your test cases
test('checks whether string is valid', () => {
    if (!helper.isValidString("string")) {
        throw new Error('String should be valid');
    }
});

test('checks array is valid', () => {
    if (!helper.isValidArray([3, 4])) {
        throw new Error('Array should be valid');
    }
});

test('checks array is invalid', () => {
    if (helper.isValidArray("notarray")) {
        throw new Error('Array should be invalid');
    }
});

test('check if this code retrieves collection from MongoDB', () => {
    const result = tester.getCollectionFn("users");
    if (!Array.isArray(result)) {
        throw new Error('Expected an array');
    }
});
