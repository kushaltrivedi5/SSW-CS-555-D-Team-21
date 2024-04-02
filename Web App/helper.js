export const isValidString = (str) => {
    return typeof str === 'string' && str.trim() !== '';
}

export const isValidArray = (arr) => {
    return Array.isArray(arr) && arr.length > 0;
}

