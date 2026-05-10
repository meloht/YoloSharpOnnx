using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    public interface IYoloTaskAsync<TResult> : IDisposable
    {
        Task<List<TResult>> RunAsync(string inputImage);

        Task<List<TResult>> RunAsync(Mat img);
    }
}
