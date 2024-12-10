import { Injectable } from '@nestjs/common';
import { loadData } from 'src/common/util/json-parser';
import { ConsumptionData } from 'src/common/energy-data.interface';

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as _ from "lodash";

@Injectable()
export class ForecastService {
    // Normalize data
    private normalizeData(
        data: ConsumptionData[]
    ): { normalized: number[]; min: number; max: number } {
        const values = data.map(d => d.value);
        const min = Math.min(...values);
        const max = Math.max(...values);

        const normalized = values.map(value => (value - min) / (max - min));
        return { normalized, min, max };
    }

    // Prepare training data (X, y)
    private prepareData(
        normalizedData: number[],
        lookBack: number
    ): { X: number[][]; y: number[] } {
        const X = [];
        const y = [];

        for (let i = 0; i < normalizedData.length - lookBack; i++) {
            X.push(normalizedData.slice(i, i + lookBack));
            y.push(normalizedData[i + lookBack]);
        }

        return { X, y };
    }

    // Build the model
    private createModel(lookBack: number): tf.Sequential {
        const model = tf.sequential();

        model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [lookBack] }));
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1 })); // Output layer
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        return model;
    }

    // Train the model
    private async trainModel(
        model: tf.Sequential,
        X: number[][],
        y: number[],
        epochs: number,
        batchSize: number
    ): Promise<tf.History> {
        const XTensor = tf.tensor2d(X);
        const yTensor = tf.tensor1d(y);

        const history = await model.fit(XTensor, yTensor, {
            epochs,
            batchSize,
            validationSplit: 0.2,
            callbacks: [tf.callbacks.earlyStopping({ patience: 3 })],
        });

        XTensor.dispose();
        yTensor.dispose();

        return history;
    }

    // Forecast daily energy data
    private forecastDaily(
        model: tf.Sequential,
        lastValues: number[],
        steps: number
    ): number[] {
        const predictions: number[] = [];
        let currentInput = lastValues.slice();

        for (let i = 0; i < steps; i++) {
            const inputTensor = tf.tensor2d([currentInput]);
            const prediction = model.predict(inputTensor) as tf.Tensor;
            const predictedValue = prediction.dataSync()[0];
            predictions.push(predictedValue);

            currentInput.shift();
            currentInput.push(predictedValue);

            inputTensor.dispose();
            prediction.dispose();
        }

        return predictions;
    }

    // Denormalize data
    private denormalizeData(
        normalizedData: number[],
        min: number,
        max: number
    ): number[] {
        return normalizedData.map(value => value * (max - min) + min);
    }

    // Calculate deviation percentage
    private calculateDeviationPercentage(
        data2022: ConsumptionData[], data2023: ConsumptionData[], values2024: number[]): { date: string; predictedValue: number; actualValue: number; deviation: string }[] {
            return data2022.map((item, index) => {
                const predictedValue = (item.value + data2023[index].value) / 2;
    
                return {
                    date: item.timestamp,
                    predictedValue,
                    actualValue: values2024[index],
                    deviation: ((Math.abs(predictedValue - values2024[index]) / values2024[index]) * 100).toFixed(2)
                };
            })
    }

    // Main execution
    async forecast() {
        const filePath2022 = 'data/input/energy-data-2022.json';
        const filePath2023 = 'data/input/energy-data-2023.json';
        const filePath2024 = 'data/input/energy-data-2024.json';

        // Load data
        const data2022 = await loadData(filePath2022);
        const data2023 = await loadData(filePath2023);
        const data2024 = await loadData(filePath2024);

        const values2024 = data2024.map(d => d.value); // Real values from 2024

        const combinedData = [...data2022, ...data2023]; // Combine 2022 and 2023 for training

        const lookBack = 30; // Days to look back

        // Normalize data
        const { normalized,  } = this.normalizeData(combinedData);

        // Prepare training data
        const { X, y } = this.prepareData(normalized, lookBack);

        // Create and train the model
        const model = this.createModel(lookBack);
        console.log('Training the model...');
        await this.trainModel(model, X, y, 50, 16);

        // Calculate deviation percentage
        const deviations = this.calculateDeviationPercentage(data2022, data2023, values2024);

        // Save results
        fs.writeFileSync('data/output/forecast.json', JSON.stringify(deviations, null, 2));

        console.log('Forecast saved');
    }
}
