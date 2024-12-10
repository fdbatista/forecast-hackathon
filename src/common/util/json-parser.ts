import * as fs from "fs";
import { ConsumptionData, DailyEnergyData } from "../energy-data.interface";

const encoding = 'utf8';

export const loadData = async (filePath: string): Promise<ConsumptionData[]> => {
    const rawData = fs.readFileSync(filePath, encoding);
    const content = JSON.parse(rawData);

    return content.entriesDaily.map((entry: DailyEnergyData) => {
        return { timestamp: entry.day, value: entry.day_total_kwh };
    });
}
