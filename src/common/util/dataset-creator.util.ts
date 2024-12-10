export const normalize = (data: number[], min: number, max: number): number[] =>
    data.map((value) => (value - min) / (max - min));

export const denormalize = (data: number[], min: number, max: number): number[] =>
    data.map((value) => value * (max - min) + min);

export const createDataset = (data: number[], lookBack: number): { X: number[][]; y: number[] } => {
    const X: number[][] = [];
    const y: number[] = [];

    for (let i = 0; i < data.length - lookBack; i++) {
        X.push(data.slice(i, i + lookBack));
        y.push(data[i + lookBack]);
    }

    return { X, y };
};
