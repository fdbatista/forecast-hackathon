import * as fs from "fs";
import * as csvParser from "csv-parser";
import { EnergyData } from "../energy-data.interface";

const filePath = 'energy-data.csv';

export const loadData = async (): Promise<number[]> => {
    const data: number[] = [];

    return new Promise((resolve, reject) => {
        fs.createReadStream(filePath)
            .pipe(csvParser())
            .on("data", (row: EnergyData) => data.push(Number(row.value_kw)))
            .on("end", () => resolve(data))
            .on("error", (error) => reject(error));
    });
};
