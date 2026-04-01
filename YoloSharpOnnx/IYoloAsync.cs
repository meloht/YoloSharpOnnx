using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx
{
    public interface IYoloAsync : IDisposable
    {
        Task<List<DetectionResult>> RunDetectAsync(string inputImage);

        Task<List<DetectionResult>> RunDetectAsync(Mat img);

        IAsyncEnumerable<DetectionBatchResult> RunDetectForeachAsync(List<string> images);
    }
}
