import * as fs from "fs";
import { EnergyData } from "../energy-data.interface";

const filePath = 'energy-data-2023-01.json';
const encoding = 'utf8';

export const loadData = async (): Promise<number[]> => {
    const rawData = fs.readFileSync(filePath, encoding);
    const jsonData: EnergyData[] = JSON.parse(rawData);
 
    return jsonData.map((entry) => entry.value_kw);
}
