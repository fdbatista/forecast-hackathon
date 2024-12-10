export interface DailyEnergyData {
    day: string;
    day_total_kwh: number;
}

export interface EnergyData {
    timestamp: string;
    value_kw: number;
}

export interface ConsumptionData {
    timestamp: string;
    value: number;
}
