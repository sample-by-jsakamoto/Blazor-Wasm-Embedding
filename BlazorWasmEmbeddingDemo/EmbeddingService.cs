using Microsoft.JSInterop;

/// <summary>
/// Provides text embedding functionality by interoperating with a JavaScript module
/// that runs a feature-extraction model in the browser.
/// </summary>
public class EmbeddingService : IAsyncDisposable
{
    private readonly Lazy<Task<IJSObjectReference>> _moduleTask;

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingService"/> class.
    /// </summary>
    public EmbeddingService(IJSRuntime js)
    {
        _moduleTask = new(() => js.InvokeAsync<IJSObjectReference>("import", "./js/embeddings.js").AsTask());
    }

    /// <summary>
    /// Initializes the embedding model. The first call may take a while since it fetches
    /// the ONNX runtime and model files from the network, but later calls complete quickly
    /// once they are cached in the browser's cache storage.
    /// </summary>
    public async Task InitAsync()
    {
        var module = await _moduleTask.Value;
        await module.InvokeVoidAsync("initModel", "Xenova/all-MiniLM-L6-v2");
    }

    /// <summary>
    /// Computes a normalized embedding vector for the given text.
    /// </summary>
    public async Task<float[]> EmbedAsync(string text)
    {
        var module = await _moduleTask.Value;
        return await module.InvokeAsync<float[]>("embed", text);
    }

    /// <summary>
    /// Releases the JavaScript module reference used by this service.
    /// </summary>
    public async ValueTask DisposeAsync()
    {
        if (_moduleTask.IsValueCreated)
        {
            var module = await _moduleTask.Value;
            try { await module.DisposeAsync(); }
            catch (JSDisconnectedException) { }
        }
    }
}
