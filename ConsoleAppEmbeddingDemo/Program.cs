var modelDir = Path.Combine(AppContext.BaseDirectory, "model");
var onnxModelPath = Path.Combine(modelDir, "model_quantized.onnx");
var vocabPath = Path.Combine(modelDir, "vocab.txt");

// Download the model and vocabulary if they don't exist
if (!Directory.Exists(modelDir)) Directory.CreateDirectory(modelDir);
if (!File.Exists(onnxModelPath)) await DownloadFileAsync("https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx", onnxModelPath);
if (!File.Exists(vocabPath)) await DownloadFileAsync("https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/vocab.txt", vocabPath);

using var embedder = new MiniLmEmbedder(onnxModelPath, vocabPath);

Console.WriteLine("Enter text to embed (or 'exit' to quit):");

while (true)
{
    Console.Write("> ");
    var text = Console.ReadLine();
    if (text == null || text == "exit") break;

    var embedding = embedder.Embed(text);

    Console.WriteLine($"Embedding length: {embedding.Length}");
    Console.WriteLine($"First 10 values: {string.Join(", ", embedding.Take(10))}");
}

static async Task DownloadFileAsync(string url, string destinationPath)
{
    Console.WriteLine($"Downloading {url}...");
    using var client = new HttpClient();
    using var stream = await client.GetStreamAsync(url);
    await using var fs = new FileStream(destinationPath, FileMode.Create, FileAccess.Write);
    await stream.CopyToAsync(fs);
}
