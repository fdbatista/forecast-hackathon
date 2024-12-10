import { Injectable } from '@nestjs/common';
import { loadData } from 'src/common/util/json-parser';
import { ConsumptionData } from 'src/common/energy-data.interface';

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as _ from "lodash";

@Injectable()
export class ForecastService {
    // Normalize data
    normalize(data: number[]): number[] {
        const min = Math.min(...data);
        const max = Math.max(...data);
        return data.map((value) => (value - min) / (max - min));
    }

    // Denormalize the data back to original scale
    denormalize(normalizedData: number[], originalData: number[]): number[] {
        const min = Math.min(...originalData);
        const max = Math.max(...originalData);
        return normalizedData.map((value) => value * (max - min) + min);
    }

    // Prepare the data for LSTM (reshape to [samples, timeSteps, features])
    prepareData(data: number[], lookBack: number): [tf.Tensor, tf.Tensor] {
        const X: number[][][] = [];
        const y: number[] = [];

        for (let i = lookBack; i < data.length; i++) {
            const inputSequence = data.slice(i - lookBack, i);
            X.push(inputSequence.map(value => [value])); // Reshaping data for LSTM
            y.push(data[i]);
        }

        return [tf.tensor3d(X, [X.length, lookBack, 1]), tf.tensor1d(y)];
    }

    createLSTMModel(lookBack: number): tf.Sequential {
        const model = tf.sequential();
        model.add(tf.layers.lstm({
            units: 50,
            activation: 'relu',
            inputShape: [lookBack, 1],
            returnSequences: false,
        }));
        model.add(tf.layers.dense({ units: 1 }));
        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
        });
        return model;
    };

    async trainLSTMModel(model: tf.Sequential, X: tf.Tensor, y: tf.Tensor): Promise<void> {
        await model.fit(X, y, {
            epochs: 50,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: [
                tf.callbacks.earlyStopping({ patience: 5 }),
            ],
        });
    };

    forecastWithLSTM(model: tf.Sequential, inputData: number[], lookBack: number, steps: number): number[] {
        let currentInput = inputData.slice(-lookBack);
        const predictions: number[] = [];

        for (let i = 0; i < steps; i++) {
            const inputTensor = tf.tensor3d([currentInput.map(value => [value])], [1, lookBack, 1]);
            const predictedValue = model.predict(inputTensor) as tf.Tensor;
            const predictedValueNum = predictedValue.dataSync()[0];

            predictions.push(predictedValueNum);
            currentInput.push(predictedValueNum);
            currentInput = currentInput.slice(1);
        }

        return predictions;
    };

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

        // Combine data from 2022 and 2023 for training
        const combinedData = [
            ...data2022.map(d => d.value),
            ...data2023.map(d => d.value),
        ];

        // Normalize the combined data
        const normalizedData = this.normalize(combinedData);

        // Prepare data for training (using a lookBack of 30 days)
        const lookBack = 10;
        const [X, y] = this.prepareData(normalizedData, lookBack);

        // Create and train the LSTM model
        const model = this.createLSTMModel(lookBack);
        await this.trainLSTMModel(model, X, y);

        // Forecast the next year (2024)
        const inputData = normalizedData.slice(-lookBack);
        const predictedNormalizedValues = this.forecastWithLSTM(model, inputData, lookBack, data2024.length);

        // Denormalize the predicted values
        const predictedValues = this.denormalize(predictedNormalizedValues, combinedData);

        console.log(predictedNormalizedValues)

        // Compare predictions with actual values for 2024 and calculate deviation
        const actualValues = data2024.map(d => d.value);
        const deviations = predictedValues.map((predicted, index) => {
            const actual = actualValues[index];
            const deviation = Math.abs((predicted - actual) / actual) * 100;
            return {
                date: data2024[index].timestamp,
                predictedValue: predicted,
                actualValue: actual,
                deviation: deviation.toFixed(2),
            };
        });

        // Output the results
        fs.writeFileSync('data/output/forecast.json', JSON.stringify(deviations, null, 2));

        console.log('Forecast saved');

        return deviations;
    }
}
