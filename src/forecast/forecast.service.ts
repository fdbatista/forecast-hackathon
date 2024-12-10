import { Injectable } from '@nestjs/common';
import { createDataset, denormalize, normalize } from 'src/common/util/dataset-creator.util';
import { loadData } from 'src/common/util/json-parser';

import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";
import * as _ from "lodash";
import * as dayjs from 'dayjs';
import { ConsumptionData } from 'src/common/energy-data.interface';

const INPUT_YEARS = [2022, 2023];
const COMPARISON_YEAR = 2024;

@Injectable()
export class ForecastService {

    async forecast() {

        // Step 1: Load and preprocess the data
        const rawData = await this.loadData();

        // Normalize the data
        const min = Math.min(...rawData);
        const max = Math.max(...rawData);
        const normalizedData = normalize(rawData, min, max);

        // Prepare training data
        const lookBack = 96; // 96 intervals = 24 hours
        const { X, y } = createDataset(normalizedData, lookBack);

        // Convert to TensorFlow tensors
        const XTensor = tf.tensor2d(X);
        const yTensor = tf.tensor1d(y);

        // Step 2: Define the model
        const model = tf.sequential();

        // Input layer and first hidden layer
        model.add(tf.layers.dense({ units: 32, inputShape: [lookBack], activation: 'relu' }));

        // Second hidden layer
        model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

        // Output layer (single neuron for regression)
        model.add(tf.layers.dense({ units: 1 }));

        // Compile the model with Adam optimizer and MSE loss
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
        });

        // Step 3: Train the model
        console.log("Training the model...");
        await model.fit(XTensor, yTensor, {
            epochs: 10,
            batchSize: 16,
            validationSplit: 0.1,
            callbacks: [
                tf.callbacks.earlyStopping({ patience: 3 }),  // Stop training if validation loss does not improve
            ],
        });

        // Step 4: Forecast the next year
        const steps = 365 * 96; // Forecast 15-minute intervals for 1 year
        console.log("Forecasting...");
        const predictions = this.forecastNextYear(model, normalizedData, lookBack, steps);

        // Step 5: Denormalize and save the predictions
        const denormalizedPredictions = denormalize(predictions, min, max);

        const realData = await this.loadFileForYear(COMPARISON_YEAR);

        const startDate = dayjs().startOf('year');

        const result = denormalizedPredictions.map((predictedValue, index) => {
            const actualValue = _.get(realData, index, 0);

            return {
                timestamp: startDate.add(index * 15, 'minute').format('YYYY-MM-DD HH:mm:ss'),
                predictedValue,
                realValue: actualValue.value,
                deviation: Math.abs(predictedValue - actualValue.value),
                errorPercentage: (Math.abs(predictedValue - actualValue.value) / actualValue.value) * 100
            }
        })

        const fileName = `data/output/predictions-${new Date().valueOf()}.json`;

        fs.writeFileSync(fileName, JSON.stringify(result, null, 2), "utf-8");

        const totalDeviationAvg = result
            .filter((entry) => Number.isFinite(entry.deviation))
            .reduce((acc, entry) => acc + entry.deviation, 0) / result.length;

        console.log(`Predictions saved to ${fileName}. Average deviation: ${totalDeviationAvg}`);

        return result;
    };

    private async loadData(): Promise<number[]> {
        const promises = INPUT_YEARS.map(this.loadFileForYear);
        const data = await Promise.all(promises);

        const [firstYear, secondYear] = data;

        const averages = []

        for (let i = 0; i < firstYear.length; i++) {
            const firstYearEntry = firstYear[i];

            const firstYearTimestamp = firstYearEntry.timestamp;
            const secondYearTimestamp = dayjs(firstYearTimestamp).add(1, 'year').format('YYYY-MM-DD HH:mm')

            const secondYearEntry = secondYear.find((entry) => entry.timestamp === secondYearTimestamp);

            let value = firstYearEntry.value;

            if (secondYearEntry) {
                value = (value + secondYearEntry.value) / 2;
            }

            averages.push(value);
        }

        return averages;
    }

    private async loadFileForYear(year: number): Promise<ConsumptionData[]> {
        const fileName = `data/input/energy-data-${year}.json`;
        return await loadData(fileName);
    }

    private forecastNextYear(
        model: tf.Sequential,
        initialData: number[],
        lookBack: number,
        steps: number
    ): number[] {
        let currentInput = initialData.slice(-lookBack); // Get the last 'lookBack' values
        const predictions: number[] = [];

        for (let i = 0; i < steps; i++) {
            // Convert the current input to a 3D tensor of shape [1, lookBack]
            const inputTensor = tf.tensor2d(currentInput, [1, lookBack]);  // Reshape the input to 2D

            // Get the prediction for the next time step
            const prediction = model.predict(inputTensor) as tf.Tensor;
            const predictedValue = prediction.dataSync()[0];

            // Store the prediction and update the input with the predicted value
            predictions.push(predictedValue);

            // Update input by sliding the window
            currentInput = currentInput.slice(1).concat(predictedValue);
        }

        return predictions;
    };

}
