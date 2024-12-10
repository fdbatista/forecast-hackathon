import { Injectable } from '@nestjs/common';
import { filePath } from 'src/common/constants';
import { loadData } from 'src/common/util/csv-parser';
import { createDataset, denormalize, normalize } from 'src/common/util/dataset-creator.util';

import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";

@Injectable()
export class ForecastService {
    
    async forecast () {
        // Step 1: Load and preprocess the data
        const rawData = await loadData(filePath);
    
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
        model.add(tf.layers.dense({ units: 50, inputShape: [lookBack] }));
        model.add(tf.layers.dense({ units: 1 }));
        model.compile({ optimizer: "adam", loss: "meanSquaredError" });
    
        // Step 3: Train the model
        console.log("Training the model...");
        await model.fit(XTensor, yTensor, {
            epochs: 20,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) =>
                    console.log(`Epoch ${epoch + 1}: Loss = ${logs?.loss}`),
            },
        });
    
        // Step 4: Forecast the next year
        const forecastNextYear = (
            model: tf.Sequential,
            initialData: number[],
            lookBack: number,
            steps: number
        ): number[] => {
            let currentInput = initialData.slice(-lookBack);
            const predictions: number[] = [];
    
            for (let i = 0; i < steps; i++) {
                const inputTensor = tf.tensor2d([currentInput], [1, lookBack]);
                const prediction = model.predict(inputTensor) as tf.Tensor;
                const predictedValue = prediction.dataSync()[0];
                predictions.push(predictedValue);
    
                // Update input by sliding window
                currentInput = [...currentInput.slice(1), predictedValue];
            }
            return predictions;
        };
    
        const steps = 365 * 96; // Forecast 15-minute intervals for 1 year
        console.log("Forecasting...");
        const predictions = forecastNextYear(model, normalizedData, lookBack, steps);
    
        // Step 5: Denormalize and save the predictions
        const denormalizedPredictions = denormalize(predictions, min, max);
        fs.writeFileSync(
            "predictions.csv",
            denormalizedPredictions.join("\n"),
            "utf-8"
        );
    
        console.log("Forecast saved to predictions.csv");
    };

}
