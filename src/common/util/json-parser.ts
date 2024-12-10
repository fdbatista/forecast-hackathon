import * as fs from "fs";
import { ConsumptionData, EnergyData } from "../energy-data.interface";

const encoding = 'utf8';

export const loadData = async (filePath: string): Promise<ConsumptionData[]> => {
    const rawData = fs.readFileSync(filePath, encoding);
    const content = JSON.parse(rawData);

    return content.entries.map((entry: EnergyData) => {
        return { timestamp: entry.timestamp, value: entry.value_kw };
    });
}
