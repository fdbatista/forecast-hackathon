export const normalize = (data: number[], min: number, max: number): number[] =>
    data.map((value) => (value - min) / (max - min));

export const denormalize = (data: number[], min: number, max: number): number[] =>
    data.map((value) => value * (max - min) + min);

export const createDataset = (data: number[], lookBack: number): { X: number[][]; y: number[] } => {
    const X: number[][] = [];
    const y: number[] = [];

    for (let i = lookBack; i < data.length; i++) {
        // Input is the previous 'lookBack' values (past days)
        const input = data.slice(i - lookBack, i);
        // Output is the next value (the one we're trying to predict)
        const output = data[i];

        X.push(input);
        y.push(output);
    }

    return { X, y };
};
