import { Injectable } from '@nestjs/common';
import { createDataset, denormalize, normalize } from 'src/common/util/dataset-creator.util';

import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";
import { loadData } from 'src/common/util/json-parser';
import { EnergyData } from 'src/common/energy-data.interface';

const INPUT_YEARS = [2022, 2023];

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

        const result: EnergyData[] = denormalizedPredictions.map((value, index) => ({
            timestamp: new Date(Date.now() + index * 15 * 60 * 1000).toISOString(),
            value_kw: value,
        }))

        fs.writeFileSync(`data/output/predictions-${new Date().valueOf()}.json`, JSON.stringify(result, null, 2), "utf-8");

        console.log("Forecast saved to predictions.json");
    };

    private async loadData(): Promise<number[]> {
        const fileNames = INPUT_YEARS.map(this.buildFilename);
        const result = []

        for (const fileName of fileNames) {
            const data = await loadData(fileName);
            result.push(...data);
        }

        return result;
    }

    private buildFilename(year: number): string {
        return `data/input/energy-data-${year}.json`;
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
