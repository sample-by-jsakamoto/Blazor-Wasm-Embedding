import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

let extractor = null;

/**
 * Initializes the feature-extraction pipeline used to compute text embeddings.
 * If the model has already been loaded, this function does nothing.
 *
 * @param {string} model - The name of the model to load for feature extraction.
 * @returns {Promise<void>} A promise that resolves once the model is ready.
 */
export async function initModel(model) {
    if (!extractor) {
        extractor = await pipeline('feature-extraction', model);
    }
}

/**
 * Computes a normalized embedding vector for the given text using the
 * initialized feature-extraction pipeline.
 *
 * @param {string} text - The input text to embed.
 * @returns {Promise<number[]>} A promise that resolves to the embedding vector as an array of numbers.
 */
export async function embed(text) {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}
