const { isValidString, isValidArray } = require('./helper');

test('checks whether string is valid', () => {
    expect(isValidString("string")).toBe(true);
});

test('checks array is valid', () => {
    expect(isValidArray([3, 4])).toBe(true);
});

test('checks array is invalid', () => {
    expect(isValidArray("notarray")).toBe(false);
});
