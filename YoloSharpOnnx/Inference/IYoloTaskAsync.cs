using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    internal interface IYoloTaskAsync<TResult, TAsyncResult> : IDisposable
    {
        Task<List<TResult>> RunAsync(string inputImage);

        Task<List<TResult>> RunAsync(Mat img);
        Task RunAsync(Mat img, Guid guid, IBatchProcessCallback<TAsyncResult> callback, Action<TAsyncResult> receiveAction);
        Task RunAsync(string inputImage, Guid guid, IBatchProcessCallback<TAsyncResult> callback, Action<TAsyncResult> receiveAction);

        Task CompleteAndCloseAsyncChannel();
    }
}
