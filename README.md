# Blazor Wasm Embedding Demo

This repository shows how to compute text embeddings (sentence vectors) in C#, both on the desktop and inside a Blazor WebAssembly app running in the browser.

## What is inside

- **ConsoleAppEmbeddingDemo**

  A .NET console app. It uses `Microsoft.ML.OnnxRuntime` and `Microsoft.ML.Tokenizers` to run the `all-MiniLM-L6-v2` ONNX model directly. Type a line of English text and it prints the first 10 values of the resulting 384 dimension embedding vector.

- **BlazorWasmEmbeddingDemo**

  A Blazor WebAssembly standalone app with the same idea. Since ONNX Runtime does not run inside a browser WebAssembly sandbox, it uses the `@xenova/transformers` JavaScript library instead, loaded from a CDN and called through JavaScript interop. Type a line of English text and click the button to see the first 10 values of the embedding vector.

Both apps use the same model (`all-MiniLM-L6-v2`) and the same pooling and normalization settings, so they produce nearly identical vectors for the same input text.

## Requirements

- .NET 10 SDK

## Running the console app

```
cd ConsoleAppEmbeddingDemo
dotnet run
```

On the first run, it downloads the ONNX model file and the tokenizer vocabulary file for `all-MiniLM-L6-v2` from Hugging Face into a local `model` folder. Later runs reuse those files.

## Running the Blazor WebAssembly app

```
cd BlazorWasmEmbeddingDemo
dotnet run
```

Open the app in a browser. The first run downloads the model files from the network, so it may take a moment. Later runs are faster because the files are cached in the browser's cache storage.

## Note on the embedding vector

Each vector is mean pooled and L2 normalized. Because of this, the cosine similarity between two vectors can be computed with a simple dot product.

## License

This project is licensed under the Unlicense. See [LICENSE.txt](LICENSE.txt) for details.
