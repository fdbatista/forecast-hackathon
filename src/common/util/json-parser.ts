import * as fs from "fs";
import { EnergyData } from "../energy-data.interface";

const encoding = 'utf8';

export const loadData = async (filePath: string): Promise<number[]> => {
    const rawData = fs.readFileSync(filePath, encoding);
    const content = JSON.parse(rawData);

    return content.entries.map((entry: EnergyData) => entry.value_kw);
}
